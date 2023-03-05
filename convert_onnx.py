import torch
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax

from vietocr.tool.config import Cfg
from vietocr.tool.translate import build_model

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
        self.model, _ = build_model(self.config)
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/vgg_transformer.yml")
    parser.add_argument("--encoder_onnx_path", type=str, default="weights/onnx/encoder.onnx")
    parser.add_argument("--decoder_onnx_path", type=str, default="weights/onnx/decoder.onnx")
    
    args = parser.parse_args()
    vietocr_onnx = VietOCRONNX(args)

if __name__ == "__main__":
    main()