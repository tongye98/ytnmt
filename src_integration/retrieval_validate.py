import torch 
import time 
import numpy as np
from torch import Tensor 
from typing import List 
import logging
import torch.nn.functional as F 
from model import Model
from data import UNK_ID, PAD_ID, BOS_ID, EOS_ID, OurData

logger = logging.getLogger(__name__)

def retrieval_search(batch_data:OurData, model:Model, cfg:dict):
    """
    Get outputs and attention scores for a given batch.
    """
    code_tokens = batch_data.code_tokens
    ast_nodes = batch_data.ast_nodes
    ast_positions = batch_data.ast_positions
    ast_edges = batch_data.ast_edges
    node_batch = batch_data.batch
    # prt = batch_data.ptr
    src_mask = (code_tokens != PAD_ID).unsqueeze(1) # src_mask (batch, 1, code_token_length)
    # src_mask: normal is True; pad is False

    with torch.no_grad():
        transformer_encoder_output, src_mask, gnn_encoder_output, node_mask = model(return_type="encode", 
                src_input_code_token=code_tokens, src_input_ast_token=ast_nodes, src_input_ast_position=ast_positions,
                node_batch=node_batch, edge_index=ast_edges, 
                trg_input=None, trg_truth=None, src_mask=src_mask, trg_mask=None)
        
    beam_size = cfg["testing"].get("beam_size", 4)
    beam_alpha = cfg["testing"].get("beam_alpha", -1)
    max_output_length = cfg["testing"].get("max_output_length", 40)
    min_output_length = cfg["testing"].get("min_output_length", 1)
    n_best = cfg["testing"].get("n_best", 1)
    return_attention = cfg["testing"].get("return_attention", True)
    return_probability = cfg["testing"].get("return_probability", True) 
    generate_unk = cfg["testing"].get("generate_unk", False)
    # repetition_penalty = cfg["testing"].get("repetition_penalty", -1)

    if beam_size < 2: 
        stacked_output, stacked_probability, stacked_attention = retrieval_greedy_search(model, transformer_encoder_output, src_mask,
                                    gnn_encoder_output, node_mask, max_output_length, min_output_length, generate_unk, 
                                    return_attention, return_probability)
    else:
        # logger.info("*"*20 + "Beam search with beam size = {}".format(beam_size) + "*"*20)
        stacked_output, stacked_probability, stacked_attention = retrieval_beam_search(model, transformer_encoder_output, src_mask, 
                                    gnn_encoder_output, node_mask, max_output_length, min_output_length, beam_size, beam_alpha, n_best,
                                    generate_unk, return_attention, return_probability)
        
    return stacked_output, stacked_probability, stacked_attention

def retrieval_greedy_search(model, transformer_encoder_output, src_mask, gnn_encoder_output, node_mask,
                  max_output_length, min_output_length, generate_unk, return_attention, return_prob):
    """
    Transformer Greedy function.
    :param: model: Transformer Model
    :param: encoder_output: [batch_size, src_len, model_dim]
    :param: src_mask: [batch_size, 1, src_len] # src_len is padded src length
    return
        - stacked_output [batch_size, steps/max_output_length]
        - stacked_scores [batch_size, steps/max_output_length] # log_softmax token probability
        - stacked_attention [batch_size, steps/max_output_length, src_len]
    """
    unk_index = UNK_ID
    pad_index = PAD_ID
    bos_index = BOS_ID
    eos_index = EOS_ID

    batch_size, _, src_length = src_mask.size()

    # start with BOS-symbol for each sentence in the batch
    generated_tokens = transformer_encoder_output.new_full((batch_size,1), bos_index, dtype=torch.long, requires_grad=False)
    # generated_tokens [batch_size, 1] generated_tokens id

    # Placeholder for scores
    generated_scores = generated_tokens.new_zeros((batch_size,1), dtype=torch.float) if return_prob is True else None
    # generated_scores [batch_size, 1]

    # Placeholder for attentions
    generated_attention_weight = generated_tokens.new_zeros((batch_size, 1, src_length), dtype=torch.float) if return_attention else None
    # generated_attention_weight [batch_size, 1, src_len]

    # a subsequent mask is intersected with this in decoder forward pass
    trg_mask = src_mask.new_ones((1, 1, 1))

    finished = src_mask.new_zeros(batch_size).byte() # [batch_size], uint8

    for step in range(max_output_length):
        with torch.no_grad():
            logits, penultimate_representation, cross_attention_weight = model(return_type="decode", 
                src_input_code_token=None, src_input_ast_token=None, src_input_ast_position=None,
                node_batch=None, edge_index=None, trg_input=generated_tokens, 
                trg_truth=None, src_mask=src_mask, node_mask=node_mask, trg_mask=trg_mask, 
                transformer_encoder_output=transformer_encoder_output,
                gnn_encoder_output=gnn_encoder_output)
            # logits [batch_size, step+1, trg_vocab_size]
            # logger.warning("logits shape = {}".format(logits.size())) [256,1, 13207]
            # logger.warning("penultation representation shape = {}".format(penultimate_representation.size())) [256,1,512]
            # assert False
            if step == 0:
                output = logits[:, -1] 
                # output [batch_size, trg_vocab_size]
                output = output.unsqueeze(1) # [batch_size, 1, trg_vocab_size]
                penultimate_representation = penultimate_representation[:, -1].unsqueeze(1)
                # [batch_size, 1, model_dim]

                log_probs, analysis, example_based_distribution = model.retriever_token(penultimate_representation, output)
                # log_probs [batch_size, 1, vocab_size]
                log_probs = log_probs.squeeze(1)
                # distribution = F.softmax(output, dim=-1)
                # log_probs = torch.log(distribution).squeeze(1)
                # logger.warning("log_probs step 0 = {}".format(log_probs.size()))
            else:
                output = logits[:, -1] #[batch_size, trg_vocab_size]
                output = output.unsqueeze(1) # [batch_size, 1, trg_vocab_size]
                current_penultimate_representation = penultimate_representation[:, -1]
                prevent_penultimate_representation = penultimate_representation[:, -2]
                mean_penultimate_representation = (current_penultimate_representation + prevent_penultimate_representation) / 2
                mean_penultimate_representation = mean_penultimate_representation.unsqueeze(1) # [batch_size, 1, model_dim]
                log_probs, analysis, example_based_distribution = model.retriever(mean_penultimate_representation, output)

                log_probs_token, analysis_token, example_based_distribution_token = model.retriever_token(current_penultimate_representation.unsqueeze(1), output)

                mixed_distribution = 0.5 * F.softmax(output.squeeze(1), dim=-1) + 0.25 * example_based_distribution + 0.25 * example_based_distribution_token
                log_probs = torch.log(mixed_distribution)

                # log_probs = log_probs.squeeze(1)
                # logger.warning("log_probs shape = {}".format(log_probs.size()))

            if not generate_unk:
                log_probs[:, unk_index] = float("-inf")
            if step < min_output_length:
                log_probs[:, eos_index] = float("-inf")

        # take the most likely token
        prob, next_words = torch.max(log_probs, dim=-1)
        # prob [batch_size]
        # next_words [batch_size]

        generated_tokens = torch.cat([generated_tokens, next_words.unsqueeze(-1)], dim=-1) # [batch_size, step+2]
        generated_scores = torch.cat([generated_scores, prob.unsqueeze(-1)], dim=-1) # [batch_size, step+2]

        if return_attention is True:
            cross_attention = cross_attention_weight.data[:, -1, :].unsqueeze(1) # [batch_size, 1, src_len]
            generated_attention_weight = torch.cat([generated_attention_weight, cross_attention], dim=1) # [batch_size, step+2, src_len]
    
        # check if previous symbol is <eos>
        is_eos = torch.eq(next_words, eos_index)
        finished += is_eos
        if (finished >= 1).sum() == batch_size:
            break

    # Remove bos-symbol
    # FIXME why need to cpu
    stacked_output = generated_tokens[:, 1:].detach().cpu().long()
    stacked_probability = generated_scores[:, 1:].detach().cpu().float() if return_prob  else None
    stacked_attention = generated_attention_weight[:, 1:, :].detach().cpu().float() if return_attention else None
    return stacked_output, stacked_probability, stacked_attention

def tile(x: Tensor, count: int, dim : int=0) -> Tensor:
    """
    Tiles x on dimension 'dim' count times. Used for beam search.
    i.e. [a,b] --count=3--> [a,a,a,b,b,b]
    :param: x [batch_size, src_len, model_dim]
    return tiled tensor
    """
    assert dim == 0
    out_size = list(x.size()) # [batch_size, src_len, model_dim]
    out_size[0] = out_size[0] * count # [batch_size*count, src_len, model_dim]
    batch_size = x.size(0)
    x = x.view(batch_size, -1).transpose(0,1).repeat(count, 1).transpose(0,1).contiguous().view(*out_size)
    return x

def retrieval_beam_search(model, transformer_encoder_output, src_mask, gnn_encoder_output, node_mask, max_output_length, min_output_length, beam_size, beam_alpha,
                n_best, generate_unk, return_attention, return_probability):
    """
    Transformer Beam Search function.
    In each decoding step, find the k most likely partial hypotheses.
    Inspired by OpenNMT-py, adapted for Transformer.
    :param: model: Transformer Model
    :param: encoder_output: [batch_size, src_len, model_dim]
    :param: src_mask: [batch_size, 1, src_len] # src_len is padded src length
    return
        - final_output [batch_size*n_best, hyp_len]
        - scores
        - attention: None 
    """
    assert beam_size > 0, "Beam size must be > 0."
    assert n_best <= beam_size, f"Can only return {beam_size} best hypotheses."

    unk_index = UNK_ID
    pad_index = PAD_ID
    bos_index = BOS_ID
    eos_index = EOS_ID
    batch_size = src_mask.size(0)

    trg_vocab_size = model.vocab_info["trg_vocab"]["size"]
    trg_mask = None 
    device = transformer_encoder_output.device

    transformer_encoder_output = tile(transformer_encoder_output.contiguous(), beam_size, dim=0)
    # encoder_output [batch_size*beam_size, src_len, model_dim] i.e. [a,a,a,b,b,b]
    src_mask = tile(src_mask, beam_size, dim=0)
    # src_mask [batch_size*beam_size, 1, src_len]
    
    gnn_encoder_output = tile(gnn_encoder_output.contiguous(), beam_size, dim=0)
    node_mask = tile(node_mask, beam_size, dim=0)

    trg_mask = src_mask.new_ones((1,1,1))

    batch_offset = torch.arange(batch_size, dtype=torch.long, device=device) # [0,1,2,... batch_size-1]
    beam_offset = torch.arange(0, batch_size*beam_size, step=beam_size, dtype=torch.long, device=device)
    # beam_offset [0,5,10,15,....] i.e. beam_size=5

    # keep track of the top beam size hypotheses to expand for each element
    # in the batch to be futher decoded (that are still "alive")
    alive_sentences = torch.full((batch_size*beam_size, 1), bos_index, dtype=torch.long, device=device)
    # alive_sentences [batch_size*beam_size, hyp_len] now is [batch_size*beam_size, 1]

    top_k_log_probs = torch.zeros(batch_size, beam_size, device=device)
    top_k_log_probs[:, 1:] = float("-inf")

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {"predictions": [[] for _ in range(batch_size)], 
                "scores": [[] for _ in range(batch_size)] }

    # Indicator if the generation is finished.
    is_finished = torch.full((batch_size, beam_size), False, dtype=torch.bool, device=device)

    for step in range(max_output_length):
        # feed the complete predicted sentences so far.
        decoder_input = alive_sentences
        with torch.no_grad():
            logits, penultimate_representation, cross_attention_weight = model(return_type="decode", 
                                src_input_code_token=None, src_input_ast_token=None, src_input_ast_position=None,
                                node_batch=None, edge_index=None, trg_input=decoder_input,
                                trg_truth=None, src_mask=src_mask, node_mask=node_mask, trg_mask=trg_mask,
                                transformer_encoder_output=transformer_encoder_output,
                                gnn_encoder_output=gnn_encoder_output)

            # for the transformer we made predictions for all time steps up to this point, so we only want to know about the last time step.
            if step == 0:
                output = logits[:, -1] # output [batch_size*beam_size, vocab_size]
                output = output.unsqueeze(1) # [batch_size*beam_size, 1, vocab_size]
                penultimate_representation = penultimate_representation[:, -1].unsqueeze(1) #[batch_size*beam_size, 1, model_dim]

                log_probs, analysis, example_based_distribution = model.retriever_token(penultimate_representation, output)
                log_probs = log_probs.squeeze(1)
                # distribution = F.softmax(output, dim=-1)
                # log_probs = torch.log(distribution).squeeze(1)
            else:
                output = logits[:,-1]
                output = output.unsqueeze(1)
                current_penultimate_representation = penultimate_representation[:, -1]
                prevent_penultimate_representation = penultimate_representation[:, -2]
                mean_penultimate_representation = (current_penultimate_representation + prevent_penultimate_representation) / 2
                mean_penultimate_representation = mean_penultimate_representation.unsqueeze(1)
                log_probs, analysis, example_based_distribution= model.retriever(mean_penultimate_representation, output)

                log_probs_token, analysis_token, example_based_distribution_token = model.retriever_token(current_penultimate_representation.unsqueeze(1), output)
                mixed_distribution = 0.5 * F.softmax(output.squeeze(1), dim=-1) + 0.25 * example_based_distribution + 0.25 * example_based_distribution_token
                # log_probs = log_probs.squeeze(1)
                log_probs = torch.log(mixed_distribution)

        # log_probs, _ = model.retriever(penultimate_representation, output)
        # log_probs = log_probs.squeeze(1)

        if not generate_unk:
            log_probs[:, unk_index] = float("-inf")
        if step < min_output_length:
            log_probs[:, eos_index] = float("-inf")

        # multiply probs by the beam probability (means add log_probs after log operation)
        log_probs += top_k_log_probs.view(-1).unsqueeze(1)
        current_scores = log_probs.clone()

        # compute length penalty
        if beam_alpha > 0:
            length_penalty = ((5.0 + (step+1)) / 6.0)**beam_alpha
            current_scores /= length_penalty
        

        # flatten log_probs into a list of possibilities
        current_scores = current_scores.reshape(-1, beam_size*trg_vocab_size)
        # current_scores [batch_size, beam_size*vocab_size]

        # pick currently best top k hypotheses
        topk_scores, topk_ids =current_scores.topk(beam_size, dim=-1)
        # topk_scores [batch_size, beam_size]
        # topk_ids [batch_size, beam_size]

        if beam_alpha > 0:
            top_k_log_probs = topk_scores * length_penalty
        else: 
            top_k_log_probs = topk_scores.clone()
        
        # Reconstruct beam origin and true word ids from flatten order
        topk_beam_index = topk_ids.div(trg_vocab_size, rounding_mode="floor")
        # topk_beam_index [batch_size, beam_size]
        topk_ids = topk_ids.fmod(trg_vocab_size) # true word ids
        # topk_ids [batch_size, beam_size]

        # map topk_beam_index to batch_index in the flat representation
        batch_index = topk_beam_index + beam_offset[:topk_ids.size(0)].unsqueeze(1)
        # batch_index [batch_size, beam_size]
        select_indices = batch_index.view(-1)
        # select_indices [batch_size*beam_size]: the number of seleced index in the batch.

        # append latest prediction
        alive_sentences = torch.cat([alive_sentences.index_select(0, select_indices), topk_ids.view(-1, 1)], dim=-1)
        # alive_sentences [batch_size*beam_size, hyp_len]

        is_finished = topk_ids.eq(eos_index) | is_finished | topk_scores.eq(-np.inf)
        # is_finished [batch_size, beam_size]
        if step + 1 == max_output_length:
            is_finished.fill_(True)
        
        # end condition is whether all beam candidates in each example are finished.
        end_condition = is_finished.all(dim=-1)
        # end_condition [batch_size]

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_sentences.view(-1, beam_size, alive_sentences.size(-1))
            # predictions [batch_size, beam_size, hyp_len]

            for sentence_idx in range(is_finished.size(0)): # look over sentences
                b = batch_offset[sentence_idx].item() # index of that example in the batch
                if end_condition[sentence_idx]:
                    is_finished[sentence_idx].fill_(True)
                
                finished_hyp = is_finished[sentence_idx].nonzero(as_tuple=False).view(-1)
                for sentence_beam_idx in finished_hyp: # look over finished beam candidates
                    number_eos = (predictions[sentence_idx, sentence_beam_idx, 1:] == eos_index).count_nonzero().item()
                    if number_eos > 1: # prediction should have already been added to the hypotheses
                        continue
                    elif (number_eos == 0 and step+1 == max_output_length) or (number_eos == 1 and predictions[sentence_idx, sentence_beam_idx, -1] == eos_index):
                        hypotheses[b].append((topk_scores[sentence_idx, sentence_beam_idx], predictions[sentence_idx, sentence_beam_idx,1:]))

                # if all n best candidates of the i-the example reached the end, save them
                if end_condition[sentence_idx]:
                    best_hyp = sorted(hypotheses[b], key=lambda x:x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break 
                        if len(pred) < max_output_length:
                            assert pred[-1] == eos_index, "Add a candidate which doesn't end with eos."
                        
                        results['scores'][b].append(score)
                        results['predictions'][b].append(pred)
            
            # batch indices of the examples which contain unfinished candidates.
            unfinished = end_condition.eq(False).nonzero(as_tuple=False).view(-1)
            # unfinished [batch_size]
            if len(unfinished) == 0:
                break
            
            # remove finished examples for the next steps.
            # shape [remaining_batch_size, beam_size]
            batch_index = batch_index.index_select(0, unfinished)
            top_k_log_probs = top_k_log_probs.index_select(0, unfinished)
            is_finished = is_finished.index_select(0, unfinished)
            batch_offset = batch_offset.index_select(0, unfinished)

            alive_sentences = predictions.index_select(0, unfinished).view(-1, alive_sentences.size(-1))

        # Reorder indices, outputs and masks
        select_indices = batch_index.view(-1)
        transformer_encoder_output = transformer_encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)
        gnn_encoder_output = gnn_encoder_output.index_select(0, select_indices)
        node_mask = node_mask.index_select(0, select_indices)

    def pad_and_stack_hyps(hyps: List[np.ndarray]):
        filled = (np.ones((len(hyps), max([h.shape[0]  for h in hyps])), dtype=int) * pad_index)
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked output
    # final_outputs [batch_size*n_best, hyp_len]
    predictions_list = [u.cpu().numpy() for r in results["predictions"] for u in r]
    final_outputs = pad_and_stack_hyps(predictions_list)
    scores = (np.array([[u.item()] for r in results['scores'] for u in r]) if return_probability else None)

    return final_outputs, scores, None


