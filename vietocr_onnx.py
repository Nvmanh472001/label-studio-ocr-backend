import torch
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax
import onnx
import onnxruntime as ort
import numpy as np

from vietocr.tool.config import Cfg
from vietocr.tool.translate import build_model, process_input, process_image

class OCREncoder(nn.Module):
    def __init__(self, model):
        super(OCREncoder, self).__init__()
        self.model = model
        
        
    def forward(self, inp):
        src = self.model.cnn(inp)
        memory = self.model.transformer.forward_encoder(src)
        return memory
    
class OCRDecoder(nn.Module):
    
    def __init__(self, model):
        super(OCRDecoder, self).__init__()
        self.model = model
        
        
    def forward(self, tgt_inp, memory):
        output, _ = self.model.transformer.forward_decoder(tgt_inp, memory)
        output = softmax(output, dim=-1)
        values, indices  = torch.topk(output, 5)
        return values, indices    



class VietOCRONNX(object):
    def __init__(self, args) -> None:
        self.agrs = args
        
        self.config = Cfg.load_config_from_file(args.config_path)
        self.model, self.vocab = build_model(self.config)
        self.model.eval()
        
        
        self.onnx_path = {
            "Encoder": args.encoder_onnx_path,
            "Decoder": args.decoder_onnx_path,
        }
        
        self.convert2onnx(self.onnx_path)
        
    def convert2onnx(self, onnx_path):
        # Convert Encoder
        encoder =  OCREncoder(self.model)
        encoder.eval()
        inp = torch.randn(5, 3, 32, 160, requires_grad=True)
        encoder_res = encoder(inp)
        print(f"Encoder Ouput Shape: {encoder_res.shape}")
        torch.onnx.export(
            encoder,
            inp,
            onnx_path["Encoder"],
            export_params=True,
            verbose=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["inp"],
            output_names=["encoder_memory"],
            dynamic_axes={
                'inp': {0: 'batch', 3: 'im_width'},
                'encoder_memory': {0: 'feat_width', 1: 'batch'}
            }
        )
        
        # Convert Decoder
        decoder = OCRDecoder(self.model)
        decoder.eval()
        
        tgt_inp = torch.randint(0, 232, (20, 1))
        memory = torch.randn(170, 1, 256, requires_grad=True)

        decoder_res = decoder(tgt_inp, memory)
        print(decoder_res)
        
        torch_triu = torch.triu
        def triu_onnx(x, diagonal=0, out=None):
            assert out is None
            assert len(x.shape) == 2 and x.size(0) == x.size(1)
            template = torch_triu(torch.ones((128, 128), dtype=torch.int32), diagonal)   #1024 is max sequence length
            mask = template[:x.size(0),:x.size(1)]
            return torch.where(mask.bool(), x, torch.zeros_like(x))
        torch.triu = triu_onnx
        
        torch.onnx.export(
            decoder,
            (tgt_inp, memory),
            onnx_path["Decoder"],
            export_params=True,
            verbose=True,
            opset_version=11,
            do_constant_folding=True,
            input_names = ['tgt_inp', 'memory'],
            output_names = ['values', 'indices'],
            dynamic_axes={
                'tgt_inp': {0: 'sequence_length', 1:'batch'}, \
                'memory': {0: 'feat_width', 1:'batch'}, \
                'values': {1: 'sequence_length', 0:'batch'}, \
                'indices': {1: 'sequence_length', 0:'batch'}
            }
        )

    def inference_onnx(self, img, max_seq_length=128, sos_token=1, eos_token=2):
        # Encoder Inference
        encoder_session = ort.InferenceSession(self.onnx_path["Encoder"], providers=['CPUExcutionProvider'])
        encoder_input = {encoder_session.get_inputs()[0]: img}
        memory = encoder_session.run(None, encoder_input)

        # Decoder Inference
        decoder_session = ort.InferenceSession(self.onnx_path["Decoder"], providers=['CPUExecutionProvider'])
        translated_sentence = [[sos_token] * len(img)]
        char_probs = [[1] * len(img)]
        max_length = 0
        
        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = np.array(translated_sentence).astype('long')
            
            values, indices = decoder_session.run(tgt_inp, memory)
            indices = indices[:, -1, 0]
            indices = indices.tolist()
            
            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)
            translated_sentence.append(indices)   
            max_length += 1

        translated_sentence = np.asarray(translated_sentence).T
        char_probs = np.asarray(char_probs).T
        
        line_probs = []
        for i in range(len(img)):
            eos_index = np.where(translated_sentence[i] == eos_token)[0][0]
            line_probs.append(np.mean(char_probs[i][:eos_index]))
        return translated_sentence, line_probs

    def help(self):
        encoder = onnx.load(self.onnx_path["Encoder"])
        decoder = onnx.load(self.onnx_path["Decoder"])
        
        # Check Onnx has valid schema
        onnx.checker.check_model(encoder)
        onnx.checker.check_model(decoder)
        
        # Print readable graph
        print(f"Encoder Graph:\n{encoder.graph}")
        print(f"Decoder Graph:\n{decoder.graph}")

    def predict(self, img, return_prob=True):
        config = self.config
        img = process_input(img, config["image_height"],
                        config["image_min_width"], config["image_max_width"])

        s, prob = self.inference_onnx(img)
        s = s[0].tolist()
        prob = prob[0].tolist()
        
        s = self.vocab.decode(s)
        
        if return_prob:
            return s, prob
        else:
            return s
        
    def predict_batch(self, img_list, return_prob=True):
        config = self.config
        total_img = len(img_list)
        
        batch_width = 0
        batch_list = []
        
        for idx, img in enumerate(img_list):
            img = process_image(img, config["image_height"],
                            config["image_min_width"], config["image_max_width"])
            img_width = img.shape[2]
            
            if img_width > batch_width:
                batch_width = img_width
            batch_list.append(img)
        
        batch = np.ones((total_img, 3, config["image_height"], batch_width))
        for idx, single_img in enumerate(batch_list):
            _, h, w = single_img.shape
            batch[idx, :, :, :w] = single_img
            
        print(batch.shape)
        
        translated_sentence, prob = self.inference_onnx(batch)
        result = []
        for idx, s in enumerate(translated_sentence):
            s = s.tolist()
            s = self.vocab.decode(s)
            if return_prob:
                result.append(s, prob[idx])
            else:
                result.append(s)
        
        return result

    def __call__(self, img, return_prob=True):
        if isinstance(img, list) and len(img) > 1:
            result = self.predict_batch(img, return_prob)
        else:
            result = self.predict(img, return_prob)
            
        return result
        
        
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/vgg_transformer.yml")
    parser.add_argument("--encoder_onnx_path", type=str, default="weights/onnx/encoder.onnx")
    parser.add_argument("--decoder_onnx_path", type=str, default="weights/onnx/decoder.onnx")
    parser.add_argument("--image_path", type=str, default="imgs/")
    
    args = parser.parse_args()
    vietocr_onnx = VietOCRONNX(args)
    
    import os
    from PIL import Image
    if os.path.isdir(args.image_path):
        fnames = os.listdir()
        fpaths = [os.path.join(args.image_path, fname) for fname in fnames]
        img = [Image.open(fpath) for fpath in fpaths]
        
    result = vietocr_onnx(img)
    print(result)
    
if __name__ == "__main__":
    main()