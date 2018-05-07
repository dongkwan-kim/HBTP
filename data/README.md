## File Naming Convention
{table_name}_{twitter_year}_{created}.{extension}


## Preprocess

### ./story/explicit-error-preprocessed
- Delete stories whose
    - writers are blocked.
    - link does not exist.
    - link does not have any contents.
- Done by hands.

### ./story/implicit-error-preprocessed
- Delete stories whose
    - link is other social media. (e.g. twitter, facebook)
    - main content is not text. (e.g. video, image, timeline)
    - length is less than 400.
- Done by hands.
 
### ./story/preprocessed
- Delete stories whose
    - link is expired.
- Delete multiple '\n' of contents.
- Delete sentences which are exact stopwords.
- Done by `../preprocess/preprocess.py`.

### ./event/synchronized
- Delete events whose stories are removed from preprocessing.
- Done by `../preprocess/synch.py`.
