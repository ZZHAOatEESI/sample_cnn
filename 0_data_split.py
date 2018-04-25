from sklearn.externals import joblib
sample_id_balanced = joblib.load('NEW_0_sample_id_balanced.pkl')
for idx, i in enumerate(sample_id_balanced):
    fw_name = 'data/' + i + '_' + sample_id_balanced[i] + '.fa'
    fw = open(fw_name, 'w+')
    if idx % 123 == 0:
        print('On sample: ', idx)
    with open('seqs_ag.fna') as f:
        n_reads = 0
        for line_idx, line in enumerate(f):
            flag = 0
            if line[0] == '>':
                n_reads += 1
                if n_reads % int(188484747/10) == 0 and idx % 123 == 0:
                    print('{0:>3}% is done'. format(int(n_reads/int(188484747/10)*10)))
                readID = line[1:].split(' ')[0]
                sampleID = readID.split('_')[0]
                if sampleID == i:
                    flag = 1
                    fw.write(line)
                else:
                    flag = 0
            else:
                if flag == 1:
                    fw.write(line)