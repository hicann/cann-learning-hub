# -*- coding: utf-8 -*-

from huaweicloud_sis.bean.vcs_register_voice_request import RegisterVoiceRequest
from huaweicloud_sis.bean.vcs_synthesis_request import VcsSynthesisRequest
from huaweicloud_sis.utils import io_utils
from huaweicloud_sis.bean.sis_config import SisConfig
from huaweicloud_sis.utils.logger_utils import logger
from huaweicloud_sis.auth import aksk_service
from huaweicloud_sis.exception.exceptions import ClientException, ServerException
import json


class VcsClient:
    """ 声音复刻client，可用于注册声音、查询已注册声音、使用已注册声音合成 """
    def __init__(self, ak, sk, region, project_id, service_endpoint=None, sis_config=None):
        """
            声音复刻client初始化
        :param ak:                  ak
        :param sk:                  sk
        :param region:              区域，如cn-north-4
        :param project_id:          项目id，可参考https://support.huaweicloud.com/api-sis/sis_03_0008.html
        :param service_endpoint:    终端节点，可不填使用默认即可
        :param sis_config:          配置信息，包括超时、代理等，可不填使用默认即可。
        """
        self._ak = ak
        self._sk = sk
        self._region = region
        self._project_id = project_id
        if service_endpoint is None:
            self._service_endpoint = 'https://sis-ext.' + region + '.myhuaweicloud.com'
        else:
            self._service_endpoint = service_endpoint
        if sis_config is None:
            self._sis_config = SisConfig()
        else:
            self._sis_config = sis_config

    def register_voice(self, request):
        """
            注册声音
        :param request: vcs注册请求
        :return: 
        """
        if not isinstance(request, RegisterVoiceRequest):
            logger.error('the parameter in \'register_voice(request)\' should be RegisterVoiceRequest class')
            raise ClientException('the parameter in \'register_voice(request)\' should be RegisterVoiceRequest class')
        url = self._service_endpoint + '/v1/' + self._project_id + '/vcs/voices'
        headers = {'Content-Type': 'application/json'}
        params = request.construct_params()
        result = aksk_service.aksk_connect(self._ak, self._sk, url, headers, params, 'POST', self._sis_config)
        return result

    def synthesis(self, request):
        """
            合成
        :param request: vcs合成请求
        :return: 
        """
        if not isinstance(request, VcsSynthesisRequest):
            logger.error('the parameter in \'synthesis_voice(request)\' should be VcsSynthesisRequest class')
            raise ClientException('the parameter in \'synthesis_voice(request)\' should be VcsSynthesisRequest class')
        url = self._service_endpoint + '/v1/' + self._project_id + '/vcs/voices/clone'
        headers = {'Content-Type': 'application/json'}
        params = request.construct_params()
        result = aksk_service.aksk_connect(self._ak, self._sk, url, headers, params, 'POST', self._sis_config)
        if 'result' not in result:
            error_msg = 'The result of tts customization is invalid. Result is %s ' % json.dumps(result)
            logger.error(error_msg)
            raise ClientException(error_msg)
        if request.get_saved():
            base_str = result['result']['data']
            io_utils.save_audio_from_base64str(base_str, request.get_saved_path())
            result['is_saved'] = True
            result['saved_path'] = request.get_saved_path()
        return result


    def query_voice_name(self, limit = 10, offset = 0):
        """
            查询声色列表
        """
        currentPageLimit = limit
        currentPageOffset = offset
        if (limit <= 0 or limit > 100):
            currentPageLimit = 100

        if offset < 0:
            currentPageOffset = 0
        url = self._service_endpoint + '/v1/' + self._project_id + '/vcs/voices' + "?limit=" + str(currentPageLimit) + "&offset=" + str(currentPageOffset)
        result = aksk_service.aksk_connect(self._ak, self._sk, url, None, None, 'GET', self._sis_config)
        return result

