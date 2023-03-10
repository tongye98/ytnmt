from data import PAD_ID, OurData
from model import Model


def search(batch_data:OurData, model:Model, cfg):
    """
    Get outputs and attention scores for a given batch.
    """
    code_tokens = batch_data.code_tokens
    ast_nodes = batch_data.ast_nodes
    text_tokens = batch_data.text_tokens
    ast_positions = batch_data.ast_positions
    ast_edges = batch_data.ast_edges
    node_batch = batch_data.batch
    # prt = batch_data.ptr 
    src_mask = (code_tokens != PAD_ID).unsqueeze(1) # src_mask (batch, 1, code_token_length)
    # src_mask: normal is True; pad is False
    text_tokens_input = text_tokens[:, :-1]
    text_tokens_output = text_tokens[:, 1:]
    # FIXME why is text_tokens output to make the trget mask
    trg_mask = (text_tokens_input != PAD_ID).unsqueeze(1) # trg_mask (batch, 1, trg_length)
    # trg_mask: normal is True, pad is False
    ntokens = (text_tokens_output != PAD_ID).data.sum().item()

    beam_size: None 
    beam_alpha: float 
    max_output_length: int
    min_output_length: int
    n_best: int
    return_attention: bool 
    return_probability: str 
    generate_unk: bool 
    repetition_penalty: float 

    return None


def greedy_search():
    return None 

def beam_search():
    return None 


if __name__ == "__main__":
    search()