class BeamSearchNode(object):
    def __init__(self, hidden, prevNode, state, logProb, length, max_len):
        self.hidden = hidden
        self.prevNode = prevNode
        self.state = state
        self.logp = logProb
        self.len = length
        self.max_len = max_len

    def eval(self, alpha=1.0):
        reward = random.random() * 1e-6
        return self.logp / float(self.len-1 + 1e-6) + alpha*reward

def beam_decode(self, trg, hiddens, outputs, input):
    topk = 1
    beam_width = 3
    decoded_batch = []

    for idx in range(trg.size(1)):
        hidden = hiddens[0][:,idx,:].unsqueeze(1)
        cell = hiddens[1][:,idx,:].unsqueeze(1)
        output = outputs[:,idx, :].unsqueeze(1)
        decoder_input = torch.LongTensor([input[idx]])

        endnodes = []
        num_required = min((topk + 1), topk - len(endnodes))
        node = BeamSearchNode((hidden,cell), None, decoder_input, 0, 1, trg.shape[0])
        node_queue = PriorityQueue()
        node_queue.put((-node.eval(), node))
        qsize = 1
        mask = self.decode_mask([output.shape[0]]).cuda()
        counter = 0

        while True:
            if qsize > 10000: break
            score, n = node_queue.get()
            decoder_input = n.state
            decoder_hidden = n.hidden
           
            if n.len == n.max_len and n.prevNode != None:
                endnodes.append((score, n))
                if len(endnodes) >= num_required:
                    break
                else:
                    continue
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, output, mask)
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = torch.LongTensor([indexes[0][new_k]])
                log_p = log_prob[0][new_k].item()
                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp+log_p, n.len+1, n.max_len)
                score = -node.eval()
                nextnodes.append((score, node))

            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                node_queue.put((score, nn))
            qsize += len(nextnodes) - 1

        if len(endnodes) == 0:
            endnodes = [node_queue.get() for _ in range(topk)]

        utterances = []
        hidden_ls = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            hidden_ls.append(n.hidden)
            utterance.append(n.state)
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.state)
                hidden_ls.append(n.hidden)
            utterance = utterance[::-1]
            hidden_ls = hidden_ls[::-1]
            utterances.append(utterance)
        
        if trg.shape[0] > len(utterances[0]):
            tmp_input = utterances[0][-1]
            tmp_hidden = hidden_ls[-1]
            tmp_utterances = []
            for t in range(trg.shape[0]-len(utterances[0])):
                output_, tmp_hidden = self.decoder(tmp_input, tmp_hidden, output, mask)
                top1 = output_.argmax(1)
                tmp_input = top1
                tmp_utterances.append(top1)
            tmp_utterances = utterances[0] + tmp_utterances
            utterances = []
            utterances.append(tmp_utterances)
        
        decoded_batch.append(utterances)
    decoded_batch = torch.LongTensor(decoded_batch)
    return decoded_batch