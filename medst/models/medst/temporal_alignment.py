import torch
import torch.nn.functional as F
from torch.autograd import Variable


def pairwise_l2_distance(embs1, embs2):
    norm1 = torch.sum(embs1**2, dim=1)
    norm1 = norm1.view(-1, 1)
    norm2 = torch.sum(embs2**2, dim=1)
    norm2 = norm2.view(1, -1)
    dist = torch.max(norm1 + norm2 - 2.0 * torch.matmul(embs1, embs2.t()), torch.tensor(0.0))
    return dist
    
def get_scaled_similarity(embs1, embs2, similarity_type, temperature):
    channels = embs1.shape[1]
    # Go from embs1 to embs2.
    if similarity_type == 'cosine':
        similarity = torch.matmul(embs1, embs2.t())
    elif similarity_type == 'l2':
        similarity = -1.0 * pairwise_l2_distance(embs1, embs2)
    else:
        raise ValueError('similarity_type can either be l2 or cosine.')

    # Scale the distance by a temperature that helps with how soft/hard the
    # alignment should be.
    similarity /= temperature
    return similarity

def align_pair_of_sequences(embs1, embs2, similarity_type,
                temperature):
    max_num_steps = embs1.shape[0]

    # Find distances between embs1 and embs2.
    sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)
    # Softmax the distance.
    softmaxed_sim_12 = F.softmax(sim_12, dim=1)

    # Calculate soft-nearest neighbors.
    nn_embs = torch.matmul(softmaxed_sim_12, embs2)

    # Find distances between nn_embs and embs1.
    sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)

    logits = sim_21
    labels = torch.eye(max_num_steps)
    # print(labels)
    return logits, labels

def first_align(embs1, embs2, similarity_type,
                temperature):
    max_num_steps = embs1.shape[0]

    # Find distances between embs1 and embs2.
    sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)
    logits = sim_12
    labels = torch.eye(max_num_steps)
    # print(labels)
    return logits, labels


def classification_loss(logits, labels):
    labels = labels.to(logits.device)
    return -torch.mean(torch.sum(Variable(labels) *
                                 F.log_softmax(logits, dim=1), dim=1), dim=0)

def regression_loss(logits, labels, num_steps, loss_type,
                    variance_lambda, huber_delta=2):
    
    # num_steps=4,
    # steps: 1234 1234
    seq_lens=None
    normalize_indices = True
    
    labels = labels.to(logits.device)
    bs = labels.shape[0]
    
    steps = (
        torch.arange(0, num_steps)
        .unsqueeze(0)
        .repeat([bs, 1])
        .to(logits.device)
    )

    # If seq_lens has not been provided assume is equal to the size of the
    # time axis in the embeddings.
    if seq_lens is None:
        seq_lens = (
            torch.tensor(num_steps)
            .unsqueeze(0)
            .repeat([bs])
            .int()
            .to(logits.device)
        )

    if normalize_indices:
        float_seq_lens = seq_lens.float()
        tile_seq_lens = (
            torch.tile(torch.unsqueeze(float_seq_lens, dim=1), [1, num_steps]) + 1e-7
        )
        steps = steps.float() / tile_seq_lens # divide seq_len
    else:
        steps = steps.float()


    beta = F.softmax(logits, dim=1)
    # print(beta.shape)  # 20 4
    true_time = torch.sum(steps * labels, dim=1)
    pred_time = torch.sum(steps * beta, dim=1)


    if loss_type in ['regression_mse', 'regression_mse_var']:
        if 'var' in loss_type:
            # Variance aware regression.
            pred_time_tiled = torch.tile(
                            torch.unsqueeze(pred_time, dim=1), [1, num_steps]
                        )

            pred_time_variance = torch.sum((steps - pred_time_tiled).pow(2) * beta, dim=1)

            # Using log of variance as it is numerically stabler.
            pred_time_log_var = torch.log(pred_time_variance + 1e-7)
            error = true_time - pred_time
            is_small_error = torch.abs(error) < huber_delta
            squared_error = (true_time - pred_time).pow(2)
            var_loss = torch.exp(-pred_time_log_var) * squared_error + variance_lambda * pred_time_log_var
            linear_loss = huber_delta * (torch.abs(error) - 0.5 * huber_delta)
            loss = torch.where(is_small_error, var_loss, linear_loss).mean()

        else:
            loss = torch.mean(F.mse_loss(true_time, pred_time))
    else:
        raise ValueError('Unsupported regression loss %s. Supported losses are: regression_mse, regression_mse_var and regression_huber.' % loss_type)

    return loss

def compute_deterministic_alignment_loss(embs1,
                                embs2,
                                num_steps,
                                loss_type,
                                similarity_type,
                                temperature,
                                variance_lambda,
                                huber_delta=None
                                ):

    labels_list = []
    logits_list = []
    steps_list = []
    seq_lens_list = []
    batch_size = embs1.shape[0]

    first_logits_list = []
    first_labels_list = []

    for i in range(batch_size):
       
        logits, labels = align_pair_of_sequences(embs1=embs1[i], embs2=embs2[i], similarity_type=similarity_type, temperature=temperature)
        logits_list.append(logits)
        labels_list.append(labels)

        logits11, labels11 = first_align(embs1=embs1[i], embs2=embs2[i], similarity_type=similarity_type, temperature=temperature)
        first_logits_list.append(logits11)
        first_labels_list.append(labels11)

            
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    logits11 = torch.cat(first_logits_list, dim=0)
    labels11 = torch.cat(first_labels_list, dim=0)


    # Forward mapping classification
    loss_classification = classification_loss(logits11, labels11)

    # Reverse mapping regression
    if 'regression' in loss_type:
        loss_reg = regression_loss(logits, labels, num_steps, loss_type, variance_lambda, huber_delta)
    else:
        raise ValueError('Unidentified loss_type %s. Currently supported loss types are: regression_mse, regression_huber, classification.' % loss_type)

    return loss_reg + loss_classification

if __name__=="__main__":

    # emb1 = torch.rand(5,4,10)
    # emb2 = torch.rand(5,4,10)
    emb1 = torch.tensor([[[0.1,0.25], [0.5,0.6],[0.8,0.9],[1.5,1.9]]])
    emb2 = torch.tensor([[[0.12,0.2], [0.55,0.65],[0.85,0.95],[1.5,1.9]]])
    loss = compute_deterministic_alignment_loss(emb1, emb2, num_steps=4, loss_type="regression_mse_var", similarity_type="cosine", temperature=0.1,variance_lambda = 0.001 )
    # print(loss)