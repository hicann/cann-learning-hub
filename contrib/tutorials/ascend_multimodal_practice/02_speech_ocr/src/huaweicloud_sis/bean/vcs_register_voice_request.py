# -*- coding: utf-8 -*-


class RegisterVoiceRequest:
    """ 声音复刻注册请求，除了初始化必选参数外，其他参数均可不配置使用默认 """
    def __init__(self, data, voice_name):
        """
            声音复刻注册请求初始化
        :param text: 需要合成的文本
        """
        self._data = data
        self._voice_name = voice_name
        self._language = 'chinese'

    def set_language(self, language):
        self._language = language

    def construct_params(self):
        config = dict()
        config['voice_name'] = self._voice_name
        config['language'] = self._language
        params = dict()
        params['data'] = self._data
        params['config'] = config
        return params
