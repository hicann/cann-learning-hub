# -*- coding: utf-8 -*-


class VcsStreamRequest:
    """ 声音复刻流式合成请求，除了初始化必选参数外，其他参数均可不配置使用默认 """
    def __init__(self, text = None):
        """
            声音复刻流式合成请求初始化
        """
        self._text = text
        self._command = 'START'
        self._audio_format = 'pcm'
        self._sample_rate = '24000'
        self._voice_name = 'chinese_huaxiaorou_common'
        self._speed = 0
        self._pitch = 0
        self._volume = 50
        self._text_pieces = 'ONLY'

    def set_command(self, command):
        self._command = command

    def set_audio_format(self, audio_format):
        self._audio_format = audio_format

    def set_sample_rate(self, sample_rate):
        self._sample_rate = sample_rate

    def set_voice_name(self, voice_name):
        self._voice_name = voice_name

    def set_speed(self, speed):
        self._speed = speed

    def set_pitch(self, pitch):
        self._pitch = pitch

    def set_volume(self, volume):
        self._volume = volume

    def set_text_pieces(self, text_pieces):
        self._text_pieces = text_pieces

    def set_text(self, text):
        self._text = text

    def construct_params(self):
        config = dict()
        config['speed'] = self._speed
        config['pitch'] = self._pitch
        config['audio_format'] = self._audio_format
        config['sample_rate'] = self._sample_rate
        config['voice_name'] = self._voice_name
        config['volume'] = self._volume
        params = dict()
        params['text_pieces'] = self._text_pieces
        params['command'] = self._command
        params['config'] = config
        if (self._text_pieces == "ONLY" and self._text is not None):
            params['text'] = self._text
        return params
