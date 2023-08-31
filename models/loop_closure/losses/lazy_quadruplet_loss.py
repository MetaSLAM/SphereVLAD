import torch


def _best_pos_distance(query, pos_vecs):
    '''
    Find the minimum distance between query feature and positives features.
    Args:
        query (tensor: batch_size x 1 x feature_dim): Output feature of query
        pos_vecs (tensor: batch_size x num_pos x feature_dim): Output feature of positives

    Returns: 
        best_pos (tensor: batch_size): Minimum distance of each query to positives.    
    '''
    if len(pos_vecs.shape) > 2:
        if len(query.shape) <= 2:
            query_copy = query.unsqueeze(1).repeat(1, pos_vecs.shape[1], 1)
            best_pos = torch.min(torch.norm(
                query_copy - pos_vecs, dim=2), dim=1)
        else:
            best_pos = torch.min(torch.norm(query - pos_vecs, dim=2), dim=1)
        best_pos = best_pos[0]
    else:
        best_pos = torch.norm(query - pos_vecs, dim=1)

    best_pos = best_pos.view(query.shape[0], -1)
    return best_pos


def _best_neg_distance(query, neg_vecs):
    """ Caculate the distance between query and negitative query
    Args:
        query    ([batch, 1, 8192]): Query information
        neg_vecs ([batch, 2, 8192]): Negative Query information
    Returns:
        [batch, 2, 1]: [distance]
    """

    if len(query.shape) > 2:
        best_neg = torch.norm(query - neg_vecs, dim=2)
    else:
        query_copy = query.unsqueeze(1).repeat(1, neg_vecs.shape[1], 1)
        best_neg = torch.norm(query_copy - neg_vecs, dim=2)

    return best_neg


class LazyTripletLoss(torch.nn.Module):
    '''
    Lazy variance of triplet loss
    '''

    def __init__(self, margin=0.2):
        super(LazyTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, q_vec, pos_vecs, neg_vecs):

        batch_size = q_vec.shape[0]
        num_neg = neg_vecs.shape[1]

        # query to positive queries
        best_pos = _best_pos_distance(q_vec, pos_vecs)
        q_pos = best_pos.reshape(-1, 1).repeat(1, num_neg)
        q_neg = _best_neg_distance(q_vec, neg_vecs)
        loss = q_pos - q_neg + self.margin
        loss = torch.clamp(loss, min=0.0)
        loss = torch.mean(loss)

        return loss


class LazySecondLoss(torch.nn.Module):
    '''
    Lazy variance of triplet loss
    '''

    def __init__(self, margin=0.2):
        super(LazySecondLoss, self).__init__()
        self.margin = margin

    def forward(self, q_vec, pos_vecs, neg_vecs, other_neg):

        batch_size = q_vec.shape[0]
        num_neg = neg_vecs.shape[1]

        # query to positive queries
        best_pos = _best_pos_distance(q_vec, pos_vecs)
        other_neg_copy = other_neg.repeat(1, neg_vecs.shape[1], 1)

        q_pos = best_pos.repeat(1, neg_vecs.shape[1])
        # print(neg_vecs.shape, other_neg_copy.shape)
        # print(best_pos.shape, q_vec.shape, pos_vecs.shape)
        q_neg = _best_neg_distance(neg_vecs, other_neg_copy)

        # print(q_pos.shape, q_neg.shape)
        loss = q_pos - q_neg + self.margin
        loss = torch.clamp(loss, min=0.0)
        loss = torch.mean(loss)

        return loss


class LazyQuadrupletLoss(torch.nn.Module):
    ''' 
    Lazy Quadruplet Loss
        Args:
        q_vec (tensor: batch_size x 1 x feature_dim): query feature
        pos_vecs (tensor: batch_size x num_pos x feature_dim): positive feature
        neg_vecs (tensor: batch_size x num_neg x feature_dim): negative feature
        other_neg (tensor: batch_size x num_neg x feature_dim): random negative feature
        m0 (float32): margin for triplet loss
        m1 (float32): margin for second loss
    Returns: quadruplet loss, triplet loss, second loss
    '''

    def __init__(self, margin_dis, margin_sec):
        super(LazyQuadrupletLoss, self).__init__()
        self.trip_loss = LazyTripletLoss(margin_dis)
        self.second_loss = LazySecondLoss(margin_sec)

    def forward(self, x):
        q_vec = x[:, 0:1, :]
        pos_vecs = x[:, 1:3, :]
        neg_vecs = x[:, 3:21, :]
        neg_other = x[:, 21:, :]

        # |q_vec - neg_vecs| >= |q_vec - pos_vecs| + margin_trans
        trip_loss = self.trip_loss(q_vec, pos_vecs, neg_vecs)
        # |q_vec - other_neg| >= |q_vec - neg_vecs| + margin_sec
        second_loss = self.second_loss(q_vec, pos_vecs, neg_vecs, neg_other)
        loss = trip_loss + second_loss

        return loss, (trip_loss, second_loss)
