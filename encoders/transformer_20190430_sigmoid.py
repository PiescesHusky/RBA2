"""
Implementation of "Attention is All You Need"
"""
import torch
import torch.nn as nn

import onmt
from onmt.encoders.encoder import EncoderBase
# from onmt.utils.misc import aeq
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.logging import logger

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = onmt.modules.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        # print('size input norm', input_norm.size()) # test
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

#latt
class LatticeEncoderLayer(nn.Module):
    """
    A single layer of the lattice encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, feat_vec_size):  # feat_vec_size added for adaptable feat_vec_size #latt
        super(LatticeEncoderLayer, self).__init__()

    #    self.self_attn = onmt.modules.MultiHeadedAttention(
    #         heads, d_model, dropout=dropout)
      # not used for RBA layer
      
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
  #latt
        self.latt_attn = onmt.modules.GlobalAttention(d_model)
        self.feat_vec_size = feat_vec_size
        self.linear_context_score = nn.Linear(feat_vec_size, 1)  # Layer for calculating context gate score
  #latt
        self.layer_norm = onmt.modules.LayerNorm(d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, values, n):   # latt   Add n to indicate nth RBA layer
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        
        #print("inputs", inputs.size())
        #inputs torch.Size([1, 9, 256])
        #inputs torch.Size([2, 17, 256])
        #print("values", values.size())
        #values torch.Size([1, 9, 768])
        #values torch.Size([2, 17, 768])
        
   #     input_norm = self.layer_norm(inputs)
        input_norm_split = torch.split(inputs, self.feat_vec_size, dim = 2)  # change 512 to 256
   #     value_norm = self.layer_norm(values)
        value_norm_split = torch.split(values, self.feat_vec_size, dim = 2)  # change 512 to 256
        

        
       # for value_norm_split1 in value_norm_split:
           # print('value b4 cat in lattice encoder in transformer', value_norm_split1.size())#test
            #value b4 cat in lattice encoder in transformer torch.Size([1, 9, 256])
            #value b4 cat in lattice encoder in transformer torch.Size([2, 17, 256])
            #                                               sent no. x max sent len x dim
        value_norm_split = torch.cat(value_norm_split, dim = 1)
        #print("value_norm_split", value_norm_split.size())#test
        #value_norm_split torch.Size([1, 27, 256])
        #value_norm_split torch.Size([2, 51, 256])
        #                           2nd dim = sent_fac1 sent_fac2 sent_fac3

  #      print('value_norm_split type', type(value_norm_split))
        for input_norm_split1 in input_norm_split:
   #         print('input in lattice encoder in transformer', input_norm_split1.size())#test
            input_norm_split2 = input_norm_split1
        query_valkey_test = zip(input_norm_split, value_norm_split)
  #      print('value in lattice encoder in transformer', value_norm_split.size())#test
      #  for value_norm_split1 in value_norm_split:
       #     print('value  individual in lattice encoder in transformer', value_norm_split1.size())#test
   #     for input_norm_split1, value_norm_split1 in query_valkey_test:
  #          print('input, value in lattice encoder in transformer', input_norm_split1.size(), value_norm_split.size())#test
        query_valkey = zip(input_norm_split, value_norm_split)
     #   context = torch.cat([self.latt_attn(input_norm_split2, value_norm_split2) for
         #                         input_norm_split2, value_norm_split2 in query_valkey], dim = 1)
         
        # logger.info("size of input", type(input_norm_split), len(input_norm_split), (input_norm_split, ))
        # logger.info("size of input 2", type(input_norm_split2), input_norm_split2)
        #print("size of input", type(input_norm_split), len(input_norm_split), (input_norm_split, ))
        #torch.set_printoptions(profile="full")
       # print("size of input 2", type(input_norm_split2), input_norm_split2)
        #torch.set_printoptions(profile="default")
        
        #print("input_norm_split2", input_norm_split2.size())
        #input_norm_split2 torch.Size([1, 9, 256])
        #input_norm_split2 torch.Size([2, 17, 256])
        #print("value_norm_split", value_norm_split.size())
        #value_norm_split torch.Size([1, 27, 256])   
        #value_norm_split torch.Size([2, 51, 256])
        # 2nd dim = sent_fac1 sent_fac2 sent_fac3
            
        num_fac = int(value_norm_split.size()[1] / input_norm_split2.size()[1])
        #print("num_fac", num_fac)
                
        input_norm_split2s = torch.split(input_norm_split2, 1, dim = 1)
        #print("input_norm_split2s", input_norm_split2s[0].size())
        #input_norm_split2s torch.Size([2, 1, 256])
        
        value_norm_splits = torch.split(value_norm_split, input_norm_split2.size()[1], dim = 1)
        #print("value_norm_splits", value_norm_splits[0].size())
        #value_norm_splits torch.Size([2, 17, 256])
        
        value_norm_splits4att = list()
        for value_norm_split_slice in value_norm_splits:
            value_norm_splits4att.append(torch.split(value_norm_split_slice, 1, dim = 1))
            #print("value_norm_splits4att[-1]", value_norm_splits4att[-1][0].size())
            #value_norm_splits4att[-1] torch.Size([2, 1, 256])

            
        value_norm_splits4atts = list()
        for i in range(input_norm_split2.size()[1]):
            value_norm_splits4att_temp = list()
            for t in range(num_fac):
                value_norm_splits4att_temp.append(value_norm_splits4att[t][i])
            value_norm_splits4att_temp = torch.cat(value_norm_splits4att_temp, 1)
            #print("value_norm_splits4att_temp", value_norm_splits4att_temp.size())
            #value_norm_splits4att_temp torch.Size([2, 3, 256])
            value_norm_splits4atts.append(value_norm_splits4att_temp)
            
        #print("len(value_norm_splits4atts), len(input_norm_split2s)", len(value_norm_splits4atts), len(input_norm_split2s))
        #len(value_norm_splits4atts), len(input_norm_split2s) 17 17
        
        contexts = list()
        for i in range(len(input_norm_split2s)):
            context_temp, _ = self.latt_attn(input_norm_split2s[i], value_norm_splits4atts[i], n, latt=True)  
            #print("context_temp", context_temp.size())
            context_scores = self.linear_context_score(input_norm_split2s[i])  # compute context scores
           # context_scores = torch.matmul(input_norm_split2s[i], self.sig_w) # compute context scores
           # context_scores = torch.unsqueeze(context_scores, 2)   
            context_gates = torch.sigmoid(context_scores)   # convert context scores to context weights
            context_temp = torch.mul(context_temp, context_gates)  # 
            contexts.append(context_temp)
            
        contexts = torch.cat(contexts, 1)
        #print("contexts", contexts.size())
        
        context = contexts

        
        #context, _ = self.latt_attn(input_norm_split2, value_norm_split, n, latt=True)   # to log if it is latt
        #print("context, _", context.size(), _.size())
        #context, _ torch.Size([1, 9, 256]) torch.Size([9, 1, 27])
        #context, _ torch.Size([2, 17, 256]) torch.Size([17, 2, 51])
        #             no. of sen x max length x dim    max length x no. of sen x ()
   #     print('type of context inputs in transformer', type(context), type(inputs))
        out = self.dropout(context) + inputs
        #out = context
        return out
        #return self.feed_forward(out)
#latt


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, embeddings_latt = False, feat_vec_size = 512):    # latt
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings

#latt
        self.embeddings_latt = embeddings_latt
        self.feat_vec_size = feat_vec_size
#latt

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])

#latt
        self.lattice = nn.ModuleList(
            [LatticeEncoderLayer(d_model, heads, d_ff, dropout, self.feat_vec_size)
             for _ in range(1)])  # change number of RBA
# number in range() must change with things below
#latt

        self.layer_norm = onmt.modules.LayerNorm(d_model)

    def forward(self, src, lengths=None, feat_merge = False):
                                                    #latt
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
     #   print('src in transformer encoder', src.size())#test
    #    print('embed in transformer encoder ', self.embeddings) #test

#latt
# check if embeddings of features is available
# if so, obtain embeddings of word and senses separately
   #     print('feat_merge in transformer', feat_merge)
        if feat_merge == 'latt':
            emb, emb_latt = self.embeddings(src)
        else:
            emb = self.embeddings(src)
#latt

   #     print('emb type in transformer encoder', type(emb)) #test
   #     print('emb in transformer encoder', emb.size()) #test
        out = emb.transpose(0, 1).contiguous()
   #     print('out in transformer encoder', out.size()) #test
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)

# out_latt initialization
        out_latt = emb_latt.transpose(0, 1).contiguous()
# out_latt initialization

        # Run the forward pass of every layer of the tranformer.
      #  for i in range(self.num_layers - 3):
 #           print(i, 'out', out) #test
    #        print('size before normal transformer', out.size(), mask.size())#test
       #     out = self.transformer[i](out, mask)

# latt included within normal transformer, interleaved, for the last three layers
     #   for i in range(3):
        #    out = self.transformer[i](out, mask)
          #  if feat_merge == 'latt':
            #    out = self.lattice[i](out, out_latt)			

# latt included within normal transformer, interleaved, for the first three layers
        #for i in range(3):
        #    out = self.transformer[i](out, mask)
            #if feat_merge == 'latt':
                #out = self.lattice[i](out, out_latt, i)
                
       #latt  test without RBA
				
 # Run the forward pass of every layer of the tranformer.
      #  for i in range(self.num_layers - 3, self.num_layers):
 #           print(i, 'out', out) #test
    #        print('size before normal transformer', out.size(), mask.size())#test
       #     out = self.transformer[i](out, mask)


        for i in range(self.num_layers - 1):
            out = self.transformer[i](out, mask)
            
        for i in range(1):
            out = self.transformer[i](out, mask)
            if feat_merge == 'latt':
                out = self.lattice[i](out, out_latt, i)			
	   
# latt interleave all layers by (transformer + new layer) * N                
     #   for i in range(self.num_layers):
       #     out = self.transformer[i](out, mask)
        #    if feat_merge == 'latt':
         #       out = self.lattice[i](out, out_latt)                
                

#latt
#  if lattice is used, run lattice layers after normal transformer encoding
      #  out_latt = emb_latt.transpose(0, 1).contiguous()
      #  if feat_merge == 'latt':
       #     for i in range(3):
      #          print('size before lattice', out.size(), out_latt.size())#test
        #        out = self.lattice[i](out, out_latt)
#latt


# additional transformer layer after lattice
   #     if feat_merge == 'latt':
    #        for i in range(2):
 #               print(i, 'out', out) #test
    #            print('size before additional normal transformer after lattice', out.size(), mask.size())#test
   #             out = self.transformer[i](out, mask)

# additional layer after lattice

        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous()
