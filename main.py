from relbert import RelBERT
import pickle
import sys

if __name__ == '__main__':
    model = RelBERT('asahi417/relbert-roberta-large')
    failcount = 0
    count = 0
    with open('./benchmarkfail/fail-0.pkl', 'rb') as f:
        pairlist = pickle.load(f)
    total = len(pairlist)
    print("Loaded " + str(total) + " tokens")
    with open('./benchmarkfail/fail-0-breakdown.txt', 'w') as f:
        for pair in pairlist:
            count += 1
            sys.stdout.write("\r Preprocessing tokens: " + str(count) + " / " + str(total))
            sys.stdout.flush()
            try:
                model.get_embedding(pair)
            except:
                failcount += 1
                f.write(pair[0] + ',' + pair[1] + '\n')

    print(failcount)