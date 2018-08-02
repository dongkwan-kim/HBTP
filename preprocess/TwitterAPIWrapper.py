import twitter
import configparser


class TwitterAPIWrapper:

    def __init__(self, config_file_path):
        self.api = self.api_twitter(config_file_path)

    def api_twitter(self, config_file_path) -> twitter.Api:
        try:
            config = configparser.ConfigParser()
            config.read(config_file_path)
            config_t = config['TWITTER']

            consumer_key = config_t['CONSUMER_KEY']
            consumer_secret = config_t['CONSUMER_SECRET']
            access_token = config_t['ACCESS_TOKEN']
            access_token_secret = config_t['ACCESS_TOKEN_SECRET']

            _api = twitter.Api(
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                access_token_key=access_token,
                access_token_secret=access_token_secret
            )
        except Exception as e:
            print('Failed to load Twitter API Configs. Do not worry, you can still use this.\n', str(e))
            _api = None

        return _api

    def verify_credentials(self):
        try:
            return self.api.VerifyCredentials()
        except Exception as e:
            return str(e)


if __name__ == '__main__':
    api = TwitterAPIWrapper('./config/config_2.ini')
    print(api.verify_credentials())
