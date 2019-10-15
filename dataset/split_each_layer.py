# encoding: utf8
dataset_nodes = {'Homo':18222,'aminer':8023, 'SacchPomb':4092, 'SacchCere':6570, 'arXiv': 14489, 'NY':102439,
                 'Cannes':438537, 'Flickr':4546,'PierreAuger':514,'YeastLandscape':4458, 'Moscow':88804}

for key in dataset_nodes:
    if key=='PierreAuger' or key=='YeastLandscape' or key=='Moscow':
        for p in [0.8]:
            fin = key+'/train'+str(p)+'.txt'
            layers = {}
            buf = open(fin,'rb').readlines()
            for line in buf:
                index, a,b,w = line.strip().split()
                if index in layers.keys():
                    layers[index].append([a,b])
                else:
                    layers[index] = [[a,b]]

            for i in layers:
                fout = open(key+'/layer_'+i+'_'+str(p)+'.txt','wb')
                for e in layers[i]:
                    fout.writelines(' '.join(e)+'\n')