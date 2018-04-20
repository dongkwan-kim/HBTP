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
    - link is expired.
    - length is less than 400.

### ./event
- No preprocessing is needed.

