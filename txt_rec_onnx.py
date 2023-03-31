import os
import torch
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax
import onnx
import onnxruntime as ort
import numpy as np

from vietocr.tool.config import Cfg
from vietocr.tool.translate import process_input, process_image, translate_onnx
from vietocr.tool.predictor import Predictor

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
        
        config = Cfg.load_config_from_name(args.model_arch)
        config['cnn']['pretrained'] = False
        config['device'] = 'cpu'
        self.config = config
         
        self.predator = Predictor(self.config)
        self.vocab = self.predator.vocab
        
        self.onnx_path = {
            "Encoder": args.encoder_onnx_path,
            "Decoder": args.decoder_onnx_path,
        }
        
        if not all([os.path.exists(onnx_file) for onnx_file in self.onnx_path.values()]):
            self.convert2onnx(self.onnx_path)

        encoder_sess, decoder_sess = self._create_infer_session()
        
        self.infer_sess = {
            "encoder_sess" : encoder_sess,
            "decoder_sess" : decoder_sess,
        }
        
        
    def convert2onnx(self, onnx_path):
        # Convert Encoder
        encoder =  OCREncoder(self.predator.model)
        encoder.eval()
        inp = torch.randn(5, 3, 32, 160, requires_grad=True)
        encoder_res = encoder(inp)
        print(f"Encoder Ouput Shape: {encoder_res.shape}")
        torch.onnx.export(
            encoder,
            inp,
            onnx_path["Encoder"],
            export_params=True,
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
        decoder = OCRDecoder(self.predator.model)
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

    
    def _create_infer_session(self):
        encoder_session = ort.InferenceSession(self.onnx_path["Encoder"], providers=ort.get_available_providers())
        decoder_session = ort.InferenceSession(self.onnx_path["Decoder"], providers=ort.get_available_providers())
        
        return encoder_session, decoder_session

    def help(self):
        encoder = onnx.load(self.onnx_path["Encoder"])
        decoder = onnx.load(self.onnx_path["Decoder"])
        
        # Check Onnx has valid schema
        onnx.checker.check_model(encoder)
        onnx.checker.check_model(decoder)
        
        # Print readable graph
        print(f"Encoder Graph:\n{encoder.graph}")
        print(f"Decoder Graph:\n{decoder.graph}")

    def preprocess_batch(self, list_img):
        config = self.config
        
        total_img = len(list_img)
        
        # Get max shape
        batch_width = 0
        batch_list = []
        for idx, img in enumerate(list_img):
            img = process_image(img, config['dataset']['image_height'], 
                    config['dataset']['image_min_width'], config['dataset']['image_max_width'])
            im_width = img.shape[2]
            if im_width > batch_width:
                batch_width = im_width
            batch_list.append(img) 
        
        # Create batch
        batch = np.ones((total_img, 3, config['dataset']['image_height'], batch_width))
        for idx, single in enumerate(batch_list):
            _, height, width = single.shape
            batch[idx, :, :, :width] = single
        return batch

    def predict(self, img):
        img = process_input(img, self.config['dataset']['image_height'], 
                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])        

        s, prob = translate_onnx(img, **self.infer_sess)
        s = s[0].tolist()
        s = self.vocab.decode(s)
        prob = prob[0]
        
        return s, prob
        
    
    def predict_batch(self, list_img):
        batch = self.preprocess_batch(list_img)
        
        translated_sentence, prob = translate_onnx(batch, **self.infer_sess)
        
        result = []
        for i, s in enumerate(translated_sentence):
            s = translated_sentence[i].tolist()
            s = self.vocab.decode(s)
            result.append((s, prob[i]))
            
        return result

    def __call__(self, img):
        if isinstance(img, list) and len(img) > 1:
            result = self.predict_batch(img)
        else:
            result = self.predict(img)
            
        return result
        
        
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch", type=str, default="vgg_transformer")
    parser.add_argument("--encoder_onnx_path", type=str, default="./onnx/ocr_encoder.onnx")
    parser.add_argument("--decoder_onnx_path", type=str, default="./onnx/ocr_decoder.onnx")
    parser.add_argument("--image_path", type=str, default="imgs/")
    
    args = parser.parse_args()
    vietocr_onnx = VietOCRONNX(args)
    
    import os
    from PIL import Image
    if os.path.isdir(args.image_path):
        fnames = os.listdir(args.image_path)
        fpaths = [os.path.join(args.image_path, fname) for fname in fnames if fname.endswith("png")]
        img = [Image.open(fpath) for fpath in fpaths]
    
    result = vietocr_onnx(img)
    print(result)
    
if __name__ == "__main__":
    main()