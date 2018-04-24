## File Naming Convention
{table_name}_{twitter_year}_{created}.{extension}


## Preprocess

### ./story/explicit-error-preprocessed
- Delete stories whose
    - writers are blocked.
    - link does not exist.
    - link does not have any contents.

### ./story/implicit-error-preprocessed
- Delete stories whose
    - link is other social media. (e.g. twitter, facebook)
    - main content is not text. (e.g. video, image, timeline)
    - length is less than 400.
    
### ./story/preprocessed
- Delete stories whose
    - link is expired.
- Delete multiple '\n' of contents.
- Delete sentences which are exact stopwords.

### ./event/synchronized
- Delete events whose stories are removed from preprocessing.
