import numpy as np
import pandas as pd
import time
from itertools import combinations
import re
import multiprocessing
import operator


def GenerateDataFrame(csvPath, threshold):    
    data = pd.read_csv(csvPath, delimiter=',', index_col=None, header=0).drop(columns=['Unnamed: 0'])

    return data.loc[:, (data==0).mean() < (1 - threshold)]

def ProbAnaliseParallel(data, originalData, filePath, columnStart, columnEnd, count, sliceIndex, conf=0.1, sup=0.1):
    print(multiprocessing.current_process())
    generatedDataFrame = data.loc[:, :data.columns[0]]
    pd.options.mode.chained_assignment = None
    startTime = time.time()
    confThreshold = conf
    supThreshold = sup
    
    if count == 1:
        iterateColumns = data.columns[columnStart:columnEnd]
    else:
        iterateColumns = data.columns
    if len(iterateColumns) == 0:
        print( 'processo: ' + str(multiprocessing.current_process()) + ', retornou, count: ' + str(count))
        return
    
    with open(filePath, "w+") as f:
        checkedItemSets = []
        itemSet = []
        for k in np.nditer(iterateColumns, ['refs_ok']):
            indexK = str(k)
            kIds = indexK.split('&')
            kIds.sort()
            if kIds not in checkedItemSets:
                checkedItemSets.append(kIds)
                for i in np.nditer(originalData.columns, ['refs_ok']):
                    indexI = str(i)
                    if indexI not in kIds:
                        columnName = indexI + '&' + indexK
                        newItemSet = columnName.split('&')
                        newItemSet.sort()
                        if newItemSet not in itemSet:
                            itemSet.append(newItemSet)
                            lineOrganizer = {}
                            column = data[indexK] * originalData[indexI]
                            numerator = float(column.sum())
                            supKI = numerator / data.shape[0]
                            if supKI >= supThreshold:
                                if count > 1:
                                    itemSetIds = columnName.split('&')
                                    for j in range(1, len(itemSetIds)):
                                        for id in combinations(itemSetIds, j):
                                            itemSetIdsDup = columnName.split('&')
                                            for i in list(id):
                                                if i in itemSetIdsDup:
                                                    itemSetIdsDup.remove(i)
                                            otherIds = '&'.join(itemSetIdsDup)
                                            confId_column_id = numerator / originalData.loc[:, itemSetIdsDup].prod(axis=1, skipna=False).sum()
                                            if confId_column_id >= confThreshold:
                                                lineOrganizer['{:<20}\t{:<16}\t{:<}'.format('&'.join(itemSetIdsDup), '&'.join(id), str(supKI))] = confId_column_id
                                            else:
                                                if len(id) == 1:
                                                    itemSetIds.remove('&'.join(id))
                                else: 
                                    lineOrganizer['{:<}\t{:<}\t{:<}'.format(str(indexK), str(indexI), str(supKI))] = numerator/originalData[indexK].sum()
                                generatedDataFrame.loc[:, columnName] = column
                                organizedLines = sorted(lineOrganizer.items(), key=operator.itemgetter(1))
                                for key, value in organizedLines:
                                    f.write('{:<}\t{:<}\n'.format(key, str(value)))
    del generatedDataFrame[generatedDataFrame.columns[0]]
    if len(generatedDataFrame.columns) > 0:
        count += 1
        filePath = filePath.replace('.', '-' + str(count) + '.')
        #print('Terminou e vai chamar de novo, tamanho da matriz recebida: ' + str(len(iterateColumns)) + ', count: ' + str(count) + ', tempo decorrido: ' + str(time.time() - startTime) + ', processo: ' + str(multiprocessing.current_process()) + ', colunas para prox iteração: ' + str(len(generatedDataFrame.columns)) + ', fatia: ' + str(sliceIndex) + '\n')
        sliceSize = len(generatedDataFrame.columns) /1
        sliceRest = len(generatedDataFrame.columns) %1
        if sliceIndex == 1:
            ProbAnaliseParallel(generatedDataFrame, originalData, filePath, sliceSize * (sliceIndex - 1), (sliceSize * sliceIndex) + sliceRest, count, sliceIndex)
        else:
            ProbAnaliseParallel(generatedDataFrame, originalData, filePath, sliceSize * (sliceIndex - 1), (sliceSize * sliceIndex), count, sliceIndex)
    else:
        return

if __name__ == '__main__':
    from multiprocessing import Process, Pool
    import time

    print('inicio')

    confs = [0.01]
    sup = 0.004

    df = GenerateDataFrame('market_bascket.csv', sup)

    tempos = []
    for k in [4, 2, 1]:
        pool = Pool(k)
        print('core: '+str(k))
        for i in range(3): 
            inicio = time.time()
            slice = round(df.shape[1]/k)
            args_list = []
            for ix, j in enumerate(range(k-1)):
                args_list.append([df, df, "teste.txt", slice*j, slice*(j+1), 1, ix+1, conf, sup])
            args_list.append([df, df, "teste.txt", slice*(j+1), df.shape[1], 1, ix+2, conf, sup])
            print(len(args_list))
            resultados = pool.starmap(ProbAnaliseParallel, args_list)
            demora = time.time() - inicio
            tempos.append(demora)

      dfr = pd.Series(tempos)
      print('Procs: {}, Confiança: {}, Suporte: {}'.format(4, conf, sup) + '; media: ' + str(dfr.mean()) + ', desvio padrao: ' + str(dfr.std())
