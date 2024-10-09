import os
import sys
import warnings

warnings.filterwarnings("ignore")

from funasr import AutoModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import chinese2digits as c2d

digit_num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class ASR:
    def __init__(self,
                 model_path,
                 vad_model_path="",
                 punc_model_path="",
                 batch_size=40,
                 is_enhance_audio=False,
                 is_denoise_audio=False,
                 device="cuda:0"):
        """
        model_path: asr主模型文件路径
        vad_model_path: 端点检测模型文件路径，可选参数
        punc_model_path: 标点模型文件路径，可以预测文本中的标点
        batch_size: 单次识别的数量，默认为20
        is_enhance_audio: 是否进行语音增强，默认为False
        is_denoise_audio: 是否进行语音降噪，默认为False
        device: 模型推理的设备，默认为cuda:0
        """
        self.model_path = model_path
        self.vad_model_path = vad_model_path
        self.punc_model_path = punc_model_path
        self.batch_size = batch_size
        self.is_enhance_audio = is_enhance_audio
        self.is_denoise_audio = is_denoise_audio
        self.device = device
        self.model = AutoModel(
            model=self.model_path,
            device=self.device,
            disable_update=True
        )
        self.punc_model = AutoModel(
            model=self.punc_model_path,
            device=self.device,
            disable_update=True
        )

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

    def convert_chinese_to_digits(self, item):
        """
        将文本中的 中文数字 转化为 阿拉伯数字
        :param item:
        :return:
        """
        text = item['text']
        key = item['key']
        converted_text = c2d.takeNumberFromString(text)['replacedText']
        converted_text = list(converted_text)
        for i in range(len(converted_text)):
            if converted_text[i] == "1" and converted_text[i + 1] not in digit_num:
                converted_text[i] = "一"
        converted_text = "".join(converted_text)
        return {'key': key, 'text': converted_text}

    def add_punctuation(self, item):
        """
        为 文本添加标点
        :param item:
        :return:
        """
        text = item['text']
        key = item['key']
        if text:
            punc_text = self.punc_model.generate(input=text)[0]['text']
        else:
            punc_text = ""
        return {"key": key, "text": punc_text}

    def generate(self, input_files=None):
        """
        model生成函数
        :param input_files:
        :return:
        """
        if 'paraformer' in self.model_path:
            res = self.model.generate(
                input=input_files,
                language="zh",
                use_itn=True,
                batch_size=len(input_files) if self.batch_size > len(input_files) else self.batch_size
            )
            res = list(map(self.add_punctuation, res))
            res = list(map(self.convert_chinese_to_digits, res))
            return res
        else:
            res = self.model.generate(
                input=input_files,
                language="zh",
                use_itn=True,
                batch_size=len(input_files) if self.batch_size > len(input_files) else self.batch_size
            )
            # todo: 判断res结果中是否所有的text均包含文本，如果text为空，调用声纹识别模型，进行识别并返回对应结果
            res = list(map(self.convert_chinese_to_digits, res))
            return res

    def transcribe(self, audio):
        """
        将音频识别为文本
        :param audio: 音频文件，格式为列表
        :return:
        """
        res_ls = []
        try:
            # 当batch size大于音频列表的长度时，将整个音频列表喂入模型
            if self.batch_size >= len(audio):
                res = self.generate(audio)
                res_ls.append(res)
            # 当batch size小于音频列表长度时，以batch size为step，将音频列表进行切割，分批次喂入模型
            else:
                batch = [audio[i:i + self.batch_size] for i in range(0, len(audio), self.batch_size)]
                for input_files in batch:
                    res = self.generate(input_files)
                    res_ls.append(res)
        except Exception as e:
            print(str(e))
        return res_ls