from format_story import *
import random
from copy import deepcopy


DATA_PATH = '../data'
STORY_PATH = os.path.join(DATA_PATH, 'story', 'preprocessed-label')


class SplitStory:

    def __init__(self, story_ratio_for_test=0.3, cnt_ratio_for_test=0.3, force_save=False):
        self.story_all = get_formatted_stories()

        self.story_test = self.story_all.clone_with_only_mapping()
        self.story_train = self.story_all.clone_with_only_mapping()

        self.already_split = False
        self.story_ratio_for_test = story_ratio_for_test
        self.cnt_ratio_for_test = cnt_ratio_for_test
        self.force_save = force_save

    def dump(self):
        file_name = 'SplitStory_{0}_{1}.pkl'.format(
            self.story_ratio_for_test,
            self.cnt_ratio_for_test,
        )
        with open(os.path.join(STORY_PATH, file_name), 'wb') as f:
            self.story_all.clear_lambda()
            self.story_test.clear_lambda()
            self.story_train.clear_lambda()
            pickle.dump(self, f)
        print('Dumped: {0}'.format(file_name))

    def load(self):
        file_name = 'SplitStory_{0}_{1}.pkl'.format(
            self.story_ratio_for_test,
            self.cnt_ratio_for_test,
        )
        try:
            with open(os.path.join(STORY_PATH, file_name), 'rb') as f:
                loaded = pickle.load(f)
                self.story_test = loaded.story_test
                self.story_train = loaded.story_train
            print('Loaded: {0}'.format(file_name))
            return True
        except Exception as e:
            print('Load Failed: {0}'.format(file_name))
            return False

    def get_train(self):
        print('Using train set, (Split: {0})'.format(self.already_split))
        return self.story_train

    def get_test(self):
        print('Using test set, (Split: {0})'.format(self.already_split))
        return self.story_test

    def split(self, story_ratio_for_test=None, cnt_ratio_for_test=None):

        if not self.force_save and self.load():
            return

        # Update story_ratio_for_test and cnt_ratio_for_test
        self.story_ratio_for_test = story_ratio_for_test or self.story_ratio_for_test
        self.cnt_ratio_for_test = cnt_ratio_for_test or self.cnt_ratio_for_test

        self.split_story(self.story_ratio_for_test, self.cnt_ratio_for_test)
        self.already_split = True

    def split_story(self, story_ratio_for_test, cnt_ratio_for_test):

        # Descending order of [(idx, (ids_list, cnt_list)), ...] by sum(list of cnt)
        sorted_story_idx_to_ids_cnt = sorted(
            ((idx, (ids_list, cnt_list)) for idx, (ids_list, cnt_list) in enumerate(
                zip(self.story_all.word_ids, self.story_all.word_cnt)
            )),
            key=lambda t: -sum(t[1][1])  # -sum(cnt_list)
        )

        # word_id -> cnt for train set.
        word_id_to_cnt_for_train = defaultdict(int)
        for (idx, (ids_list, cnt_list)) in sorted_story_idx_to_ids_cnt:
            for word_id, cnt in zip(ids_list, cnt_list):
                word_id_to_cnt_for_train[word_id] += cnt

        pivot = int(len(sorted_story_idx_to_ids_cnt)*story_ratio_for_test)

        # Test set will be constructed from stories which have many texts, (< pivot).
        for (idx, (ids_list, cnt_list)) in sorted_story_idx_to_ids_cnt[:pivot]:
            self.story_test.word_ids[idx] = self.story_all.word_ids[idx]
            self.story_train.word_ids[idx] = self.story_all.word_ids[idx]

            cnt_list_for_test, cnt_list_for_train = self.split_cnt(
                ids_list,
                cnt_list,
                cnt_ratio_for_test,
                word_id_to_cnt_for_train,
            )
            self.story_test.word_cnt[idx] = cnt_list_for_test
            self.story_train.word_cnt[idx] = cnt_list_for_train

        # Rest will be added to train set.
        for (idx, (ids_list, cnt_list)) in sorted_story_idx_to_ids_cnt[pivot:]:
            self.story_train.word_ids[idx] = self.story_all.word_ids[idx]
            self.story_train.word_cnt[idx] = cnt_list

    def split_cnt(self, ids_list: list, cnt_list: list, cnt_ratio_for_test: float,
                  word_id_to_cnt_for_train: dict) -> tuple:
        """
        :param ids_list: list of int (ids)
        :param cnt_list: list of int (cnt)
        :param cnt_ratio_for_test: sum(cnt_list_for_test) / sum(cnt_list)
        :param word_id_to_cnt_for_train: word_id -> cnt for train set.
        :return: tuple (cnt_list_for_test, cnt_list_for_train)
        """
        cnt_list_for_test = [0 for _ in cnt_list]
        cnt_list_for_train = deepcopy(cnt_list)
        word_index_pool = [i for i, cnt in enumerate(cnt_list) for _ in range(cnt)]
        random.shuffle(word_index_pool)

        # cnt_list = sum(te, tr) for (te, tr) in zip(cnt_list_for_test, cnt_list_for_train)
        # sum(cnt_list_for_test) : sum(cnt_list_for_train) = ratio : 1 - ratio
        sum_cnt = sum(cnt_list)
        while sum(cnt_list_for_test)/sum_cnt < cnt_ratio_for_test:
            selected_idx = word_index_pool.pop()
            selected_word_id = ids_list[selected_idx]

            # word in the test set must exist in the train set
            if word_id_to_cnt_for_train[selected_word_id] != 1:
                cnt_list_for_train[selected_idx] -= 1
                cnt_list_for_test[selected_idx] += 1
                word_id_to_cnt_for_train[selected_idx] -= 1

        return cnt_list_for_test, cnt_list_for_train


if __name__ == '__main__':
    data = SplitStory()
    data.split()
    data.dump()
