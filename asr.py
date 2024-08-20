from funasr import AutoModel


class ASR:
    def __init__(self,
                 audio,
                 model,
                 vad_model="",
                 punc_model="",
                 batch_size=1,
                 is_enhance_audio=False,
                 is_denoise_audio=False,
                 device="cuda:0"):
        """
        audio: 要进行识别的音频，格式为list列表
        model: asr主模型文件路径
        vad_model: 端点检测模型文件路径，可选参数
        punc_model: 标点模型文件路径，可以预测文本中的标点
        batch_size: 单次识别的数量，默认为1，即一次识别一条音频
        is_enhance_audio: 是否进行语音增强，默认为False
        is_denoise_audio: 是否进行语音降噪，默认为False
        device: 模型推理的设备，默认为cuda:0
        """
        self.audio = audio
        self.model = model
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.batch_size = batch_size
        self.is_enhance_audio = is_enhance_audio
        self.is_denoise_audio = is_denoise_audio
        self.device = device

    def audio_enhance(self, audio):
        """
        对语音进行增强
        """
        pass

    def audio_denoise(self, audio):
        """
        对于语音进行降噪
        """
        pass

    def transcribe(self):
        """
        对语音进行转录
        """
        model = AutoModel(
            model=self.model,
            trust_remote_code=True,
            vad_model=self.vad_model if self.vad_model != "" else None,
            vad_kwargs={"max_single_segment_time": 10000} if self.vad_model != "" else {},
            punc_model=self.punc_model if self.punc_model != "" else None,
            device=self.device
        )
        res_ls = []
        # 设置batch size
        if self.batch_size >= len(self.audio):
            res = model.generate(
                input=self.audio,
                language="zh",
                use_itn=True,
                merge_vad=True,
                merge_length_s=10
            )
            res_ls.append(res)
        else:
            batch = [self.audio[i:i + self.batch_size] for i in range(0, len(self.audio), self.batch_size)]
            for input_files in batch:
                res = model.generate(
                    input=input_files,
                    language="zh",
                    use_itn=True,
                    merge_vad=True,
                    merge_length_s=10
                )
                # todo: 判断res结果中是否所有的text均包含文本，如果text为空，调用声纹识别模型，进行识别并返回对应结果
                res_ls.append(res)
        return res_ls
