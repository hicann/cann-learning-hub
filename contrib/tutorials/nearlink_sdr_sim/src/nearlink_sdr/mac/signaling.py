"""控制面信令注册表 -- TXS-10002-2025 标准 7.3.2 / 附录 G

维护 data_type_index 到信令类型的映射, 支持自动编解码。
"""

from __future__ import annotations

__all__ = [
    "decode_signaling",
    "encode_signaling",
    "get_signaling_name",
    "list_registered",
    "register_signaling",
]


from typing import Any

from nearlink_sdr.mac.frame import ControlFrame
from nearlink_sdr.mac.link_control import (
    AsyncLinkParamRequest,
    AsyncLinkParamResponse,
    AsyncMulticastLinkSetup,
    AsyncMulticastParamExchangeRequest,
    AsyncMulticastParamExchangeResponse,
    AsyncMulticastParamUpdateIndication,
    AsyncMulticastParamUpdateRequest,
    AsyncMulticastReconfig,
    AsyncTTLinkSetup,
    AsyncUnicastUpdate,
    BroadcastHopMap5GUpdate,
    BroadcastHopMapUpdate,
    BroadcastLinkDisconnect,
    BroadcastLinkParamUpdate,
    BroadcastLinkSetup,
    Channel5GStatusIndication,
    ChannelReportConfig,
    ChannelStatusIndication,
    ClockAccuracyRequest,
    ClockAccuracyResponse,
    CoordinateConfig,
    CoordinateReport,
    CoordinateRequest,
    CrcSwitchIndication,
    CrcSwitchRequest,
    DataLengthRequest,
    DataLengthResponse,
    FeatureExchangeRequest,
    FeatureExchangeResponse,
    HopMap5GUpdate,
    HopMapUpdate,
    HopTableUpdate,
    IntervalUpdateIndication,
    IntervalUpdateRequest,
    IntervalUpdateResponse,
    IsochronousLinkSetup,
    IsochronousParamExchangeRequest,
    IsochronousParamExchangeResponse,
    IsochronousParamUpdateIndication,
    IsochronousParamUpdateRequest,
    LinkDisconnect,
    MinAvailableChannels,
    MulticastDisconnect,
    MultiIntervalUpdateIndication,
    MultiIntervalUpdateRequest,
    MultiIntervalUpdateResponse,
    NarrowbandDelayRequest,
    NarrowbandDelayResponse,
    NarrowbandFreqTable24Update,
    NarrowbandFreqTable51Update,
    NarrowbandFreqTable58Update,
    NarrowbandMeasAction,
    NarrowbandMeasCapRequest,
    NarrowbandMeasCapResponse,
    NarrowbandMeasConfig,
    NarrowbandMeasConfigUpdateIndication,
    NarrowbandMeasConfigUpdateRequest,
    NarrowbandMeasReport,
    NarrowbandProxySensingFeedback,
    NarrowbandProxySensingRequest,
    NarrowbandSensingAction,
    NarrowbandSensingCapRequest,
    NarrowbandSensingCapResponse,
    NarrowbandSensingConfig,
    NarrowbandSensingConfigFeedback,
    NarrowbandSensingFeedback,
    NarrowbandSensingReport,
    NarrowbandSensingRequest,
    PhyUpdateIndication,
    PhyUpdateRequest,
    PingRequest,
    PingResponse,
    ResourceReservation,
    ResourceReservationTerminate,
    RoleSwitchRequest,
    SecurityPauseRequest,
    SecurityPauseResponse,
    SecurityRequest,
    SecurityResponse,
    SecurityStartRequest,
    SecurityStartResponse,
    SensingDeviceStatusReport,
    SignalingReject,
    SMFParamUpdateIndication,
    SMFParamUpdateRequest,
    SMFSignalingTerminate,
    SMFTimeSlotUpdateRequest,
    SMFTimeSlotUpdateResponse,
    SystemTimeIndication,
    TimeOffsetIndication,
    TimeoutUpdateRequest,
    UnknownFeatureFeedback,
    UWBMeasAction,
    UWBMeasCapRequest,
    UWBMeasCapResponse,
    UWBMeasConfig,
    UWBMeasConfigFeedback,
    UWBMeasReport,
    UWBProxySensingFeedback,
    UWBProxySensingRequest,
    UWBSensingAction,
    UWBSensingCapRequest,
    UWBSensingCapResponse,
    UWBSensingConfig,
    UWBSensingConfigFeedback,
    UWBSensingProcessFeedback,
    UWBSensingProcessRequest,
    UWBSensingReport,
    VersionExchange,
)
from nearlink_sdr.mac.power_control import (
    PowerChangeIndication,
    PowerControlRequest,
    PowerControlResponse,
)
from nearlink_sdr.mac.security import (
    GNodeConfirmCode,
    GNodeConfirmCodeWithRandom,
    GNodeDHKeyVerify,
    PairingConfirm,
    PairingFailure,
    PairingInitialInfo,
    PairingInitiate,
    PairingRequest,
    PairingResponse,
    RaMessage,
    RbMessage,
    RgMessage,
    RtMessage,
    TNodeConfirmCode,
    TNodeConfirmCodeWithRandom,
    TNodeDHKeyVerify,
)

# ---------------------------------------------------------------------------
# 信令注册表
# ---------------------------------------------------------------------------

# data_type_index -> (名称, 信令类, 字节长度)
_SIGNALING_REGISTRY: dict[int, tuple[str, type, int]] = {
    0x0000: ("收发间隔更新请求", IntervalUpdateRequest, 1),
    0x0001: ("收发间隔更新响应", IntervalUpdateResponse, 1),
    0x0002: ("收发间隔更新指示", IntervalUpdateIndication, 8),
    0x0003: ("信令被拒指示", SignalingReject, 3),
    0x0004: ("安全请求", SecurityRequest, 13),
    0x0005: ("安全响应", SecurityResponse, 12),
    0x0006: ("安全启动请求", SecurityStartRequest, 0),
    0x0007: ("安全启动响应", SecurityStartResponse, 1),
    0x0008: ("安全暂停请求", SecurityPauseRequest, 0),
    0x0009: ("安全暂停响应", SecurityPauseResponse, 0),
    0x000A: ("特性交互请求", FeatureExchangeRequest, 10),
    0x000B: ("特性交互响应", FeatureExchangeResponse, 10),
    0x000C: ("未知特性反馈", UnknownFeatureFeedback, 2),
    0x000D: ("版本交互指示", VersionExchange, 5),
    0x000E: ("数据长度请求", DataLengthRequest, 8),
    0x000F: ("数据长度响应", DataLengthResponse, 8),
    0x0010: ("信道上报指示", ChannelReportConfig, 3),
    0x0011: ("信道状态指示", ChannelStatusIndication, 20),
    0x0012: ("跳频表更新指示", HopTableUpdate, 5),
    0x0013: ("跳频地图更新指示", HopMapUpdate, 14),
    0x0014: ("最少可用信道指示", MinAvailableChannels, 2),
    0x0015: ("CRC切换请求", CrcSwitchRequest, 12),
    0x0016: ("CRC切换指示", CrcSwitchIndication, 16),
    0x0017: ("物理层更新请求", PhyUpdateRequest, 4),
    0x0018: ("物理层更新指示", PhyUpdateIndication, 8),
    0x0019: ("功率控制请求", PowerControlRequest, 3),
    0x001A: ("功率控制响应", PowerControlResponse, 4),
    0x001B: ("功率变化指示", PowerChangeIndication, 4),
    0x001C: ("时钟精度请求", ClockAccuracyRequest, 1),
    0x001D: ("时钟精度响应", ClockAccuracyResponse, 1),
    0x001E: ("链路断开指示", LinkDisconnect, 4),
    0x001F: ("异步组播链路参数重配置指示", AsyncMulticastReconfig, 31),
    0x0020: ("链接态异步链路参数更新请求", AsyncLinkParamRequest, 27),
    0x0021: ("链接态异步链路参数更新响应", AsyncLinkParamResponse, 27),
    0x0022: ("同步等时链路建链指示", IsochronousLinkSetup, 56),
    0x0023: ("同步等时链路参数交互请求", IsochronousParamExchangeRequest, 52),
    0x0024: ("同步等时链路参数交互响应", IsochronousParamExchangeResponse, 52),
    0x0025: ("同步等时链路参数更新请求", IsochronousParamUpdateRequest, 3),
    0x0026: ("同步等时链路参数更新指示", IsochronousParamUpdateIndication, 9),
    0x0027: ("链接态广播链路建立指示", BroadcastLinkSetup, 44),
    0x0028: ("广播链路参数更新指示", BroadcastLinkParamUpdate, 32),
    0x0029: ("广播链路跳频地图更新指示", BroadcastHopMapUpdate, 14),
    0x002A: ("广播链路断开指示", BroadcastLinkDisconnect, 5),
    0x002B: ("系统管理帧参数更新请求", SMFParamUpdateRequest, 8),
    0x002C: ("系统管理帧参数更新指示", SMFParamUpdateIndication, 12),
    0x002D: ("系统管理帧时间片更新请求", SMFTimeSlotUpdateRequest, 13),
    0x002E: ("系统管理帧时间片更新响应", SMFTimeSlotUpdateResponse, 9),
    0x0030: ("系统管理帧信令传输终止", SMFSignalingTerminate, 1),
    0x0031: ("角色切换请求", RoleSwitchRequest, 4),
    0x0032: ("时间偏移指示", TimeOffsetIndication, 8),
    0x0033: ("PING请求", PingRequest, 0),
    0x0034: ("PING响应", PingResponse, 0),
    0x0035: ("5G信道状态指示", Channel5GStatusIndication, 50),
    0x0036: ("5G跳频地图更新", HopMap5GUpdate, 29),
    0x0037: ("多间隔更新请求", MultiIntervalUpdateRequest, 31),
    0x0038: ("多间隔更新响应", MultiIntervalUpdateResponse, 31),
    0x0039: ("多间隔更新指示", MultiIntervalUpdateIndication, 36),
    0x003A: ("链接态单播异步链路参数更新指示", AsyncUnicastUpdate, 15),
    0x003B: ("广播链路5G跳频地图更新", BroadcastHopMap5GUpdate, 29),
    0x003C: ("超时时间更新请求", TimeoutUpdateRequest, 2),
    0x003D: ("组播链路断开指示", MulticastDisconnect, 3),
    0x003E: ("系统时间指示", SystemTimeIndication, 0),
    0x003F: ("异步组播链路建链指示", AsyncMulticastLinkSetup, 45),
    0x0040: ("异步组播参数交互请求", AsyncMulticastParamExchangeRequest, 41),
    0x0041: ("异步组播参数交互响应", AsyncMulticastParamExchangeResponse, 41),
    0x0042: ("异步组播参数更新请求", AsyncMulticastParamUpdateRequest, 3),
    0x0043: ("异步组播参数更新指示", AsyncMulticastParamUpdateIndication, 9),
    0x0044: ("窄带测量能力请求", NarrowbandMeasCapRequest, 0),
    0x0045: ("窄带测量能力响应", NarrowbandMeasCapResponse, 32),
    0x0046: ("窄带频率表2.4G更新", NarrowbandFreqTable24Update, 11),
    0x0047: ("窄带频率表5.1G更新", NarrowbandFreqTable51Update, 26),
    0x0048: ("窄带频率表5.8G更新", NarrowbandFreqTable58Update, 17),
    0x0049: ("窄带测量配置", NarrowbandMeasConfig, 0),
    0x004A: ("窄带测量报告", NarrowbandMeasReport, 0),
    0x004B: ("窄带测量动作", NarrowbandMeasAction, 6),
    0x004C: ("坐标请求", CoordinateRequest, 0),
    0x004D: ("坐标报告", CoordinateReport, 24),
    0x004E: ("坐标配置", CoordinateConfig, 24),
    0x004F: ("窄带时延请求", NarrowbandDelayRequest, 0),
    0x0050: ("窄带时延响应", NarrowbandDelayResponse, 0),
    0x0051: ("异步TT链路建链指示", AsyncTTLinkSetup, 0),
    0x0052: ("UWB测量能力请求", UWBMeasCapRequest, 0),
    0x0053: ("UWB测量能力响应", UWBMeasCapResponse, 50),
    0x0054: ("UWB测量配置", UWBMeasConfig, 0),
    0x0055: ("UWB测量配置反馈", UWBMeasConfigFeedback, 2),
    0x0056: ("UWB测量报告", UWBMeasReport, 0),
    0x0057: ("UWB感知能力请求", UWBSensingCapRequest, 0),
    0x0058: ("UWB感知能力响应", UWBSensingCapResponse, 51),
    0x0059: ("UWB感知配置", UWBSensingConfig, 0),
    0x005A: ("UWB感知配置反馈", UWBSensingConfigFeedback, 2),
    0x005B: ("UWB感知报告", UWBSensingReport, 0),
    0x005C: ("UWB感知动作", UWBSensingAction, 6),
    0x005D: ("资源预留指示", ResourceReservation, 13),
    0x005E: ("资源预留终止", ResourceReservationTerminate, 2),
    0x005F: ("窄带感知请求", NarrowbandSensingRequest, 16),
    0x0060: ("窄带感知反馈", NarrowbandSensingFeedback, 2),
    0x0061: ("窄带代理感知请求", NarrowbandProxySensingRequest, 11),
    0x0062: ("窄带代理感知反馈", NarrowbandProxySensingFeedback, 14),
    0x0063: ("窄带感知能力请求", NarrowbandSensingCapRequest, 0),
    0x0064: ("窄带感知能力响应", NarrowbandSensingCapResponse, 50),
    0x0065: ("窄带感知配置", NarrowbandSensingConfig, 0),
    0x0066: ("窄带感知配置反馈", NarrowbandSensingConfigFeedback, 2),
    0x0067: ("感知设备状态报告", SensingDeviceStatusReport, 2),
    0x0068: ("窄带感知报告", NarrowbandSensingReport, 0),
    0x0069: ("窄带感知动作", NarrowbandSensingAction, 6),
    0x006A: ("窄带测量配置更新请求", NarrowbandMeasConfigUpdateRequest, 32),
    0x006B: ("窄带测量配置更新指示", NarrowbandMeasConfigUpdateIndication, 32),
    0x006C: ("UWB感知处理请求", UWBSensingProcessRequest, 16),
    0x006D: ("UWB感知处理反馈", UWBSensingProcessFeedback, 2),
    0x006E: ("UWB代理感知请求", UWBProxySensingRequest, 11),
    0x006F: ("UWB代理感知反馈", UWBProxySensingFeedback, 14),
    0x0070: ("UWB测量动作", UWBMeasAction, 6),
    # 9.2 配对与鉴权信令
    0x0133: ("配对发起", PairingInitiate, 1),
    0x0134: ("配对请求", PairingRequest, 10),
    0x0135: ("配对回应", PairingResponse, 10),
    0x0136: ("配对确认", PairingConfirm, 70),
    0x0137: ("配对初始信息", PairingInitialInfo, 64),
    0x0138: ("T节点确认码", TNodeConfirmCode, 16),
    0x0139: ("Ra消息", RaMessage, 16),
    0x013A: ("Rb消息", RbMessage, 16),
    0x013B: ("G节点确认码与随机数", GNodeConfirmCodeWithRandom, 32),
    0x013C: ("T节点确认码与随机数", TNodeConfirmCodeWithRandom, 32),
    0x013F: ("G节点确认码", GNodeConfirmCode, 16),
    0x0141: ("G节点DH Key验证码", GNodeDHKeyVerify, 16),
    0x0142: ("T节点DH Key验证码", TNodeDHKeyVerify, 16),
    0x0147: ("配对失败", PairingFailure, 1),
    0x0148: ("Rg消息", RgMessage, 64),
    0x0149: ("Rt消息", RtMessage, 64),
}


def register_signaling(data_type_index: int, name: str, cls: type, byte_length: int) -> None:
    """注册一个新的信令类型。

    信令类必须实现 pack() -> bytes 和 unpack(bytes) -> Self 方法。
    """
    _SIGNALING_REGISTRY[data_type_index] = (name, cls, byte_length)


def encode_signaling(msg: Any) -> ControlFrame:
    """将信令消息编码为控制面帧。"""
    data_type_index = msg.DATA_TYPE_INDEX
    payload = msg.pack()
    return ControlFrame(data_type_index, payload)


def decode_signaling(frame: ControlFrame) -> Any:
    """将控制面帧解码为信令消息。

    如果 data_type_index 未注册, 返回原始 ControlFrame。
    """
    entry = _SIGNALING_REGISTRY.get(frame.data_type_index)
    if entry is None:
        return frame
    _name, cls, _byte_len = entry
    return cls.unpack(frame.payload)


def get_signaling_name(data_type_index: int) -> str:
    """获取信令名称, 未注册则返回 '未知'。"""
    entry = _SIGNALING_REGISTRY.get(data_type_index)
    return entry[0] if entry else "未知"


def list_registered() -> list[tuple[int, str, int]]:
    """列出所有已注册的信令类型: [(index, name, byte_length), ...]"""
    return [(idx, name, blen) for idx, (name, _cls, blen) in sorted(_SIGNALING_REGISTRY.items())]
