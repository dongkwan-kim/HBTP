from wordcloud import WordCloud
import matplotlib.pyplot as plt
import palettable as pal
import random
import numpy as np
from PIL import Image
import os


# Kim, Jooyeon, et al. "Leveraging the crowd to detect and reduce the spread of fake news and misinformation."
# Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. ACM, 2018.
SAMPLE_TEXT = 'Online social networking sites are experimenting with the following crowd-powered procedure to reduce ' \
              'the spread of fake news and misinformation: whenever a user is exposed to a story through her feed, ' \
              'she can flag the story as misinformation and, if the story receives enough flags, it is sent to a ' \
              'trusted third party for fact checking. If this party identifies the story as misinformation, ' \
              'it is marked as disputed. However, given the uncertain number of exposures, the high cost of fact ' \
              'checking, and the trade-off between flags and exposures, the above mentioned procedure requires ' \
              'careful reasoning and smart algorithms which, to the best of our knowledge, do not exist to date. '


CURRENT_DIR = os.path.dirname(__file__)


# https://jiffyclub.github.io/palettable/
def get_color_func(color_scheme=pal.cmocean.sequential.Algae_15):

    def _color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        len_colors = len(color_scheme.colors)
        return tuple(color_scheme.colors[random.randint(0, len_colors - 1)])

    return _color_func


def get_mask(file_name='circle.png'):
    return np.array(Image.open(os.path.join(CURRENT_DIR, file_name)))


def draw_word_cloud(text, color_func):

    word_cloud = WordCloud(
        width=600,
        height=600,
        background_color='white',
        mask=get_mask(),
    )

    word_cloud.generate(text)
    word_cloud.recolor(color_func=color_func, random_state=3)

    plt.figure()
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    draw_word_cloud(SAMPLE_TEXT, get_color_func())
