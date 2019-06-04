import pytube
import os
import pandas as pd
import multiprocessing


def multi_thread(index):
    path = "D:/DATASET/kinetics_train/"
    filename = "kinetics_train.csv"
    file = pd.read_csv(path + filename, sep=",")
    video_path = "D:/DATASET/Kinetics/Video/"
    row = file.loc[index]
    # print(row)
    label, youtube_id = row['label'].strip(), row['youtube_id'].strip()


    # print(str(index)+", "+label, youtube_id)
    format_list = ['.webm', '.mp4']
    exist = False
    for file_format in format_list:
        if os.path.isfile(video_path+label+'/'+label+'_'+str(index)+file_format):
            exist = True
            # print(label + '_' + str(index) + " already exists!!")
            continue

    if not exist:
        '''
        fss = yt.streams.filter(progressive=True).all() #order_by('res')
        print(yt.streams.first())
        for fs in fss:
            print(fs)
        '''

        try:
            yt = pytube.YouTube("https://www.youtube.com/watch?v=" + youtube_id)
            yt.streams.filter(progressive=True).desc().first().\
                download(video_path+label+'/', filename_prefix=label+"_", filename=str(index))
            print(label+"_"+str(index)+" 다운완료")
        except pytube.exceptions.VideoUnavailable:
            print(youtube_id + " 동영상 존재하지 않음")
            pass

if __name__=="__main__":
    path = "D:/DATASET/kinetics_train/"
    filename = "kinetics_train.csv"
    file = pd.read_csv(path + filename, sep=",")
    labels = file.drop_duplicates()['label'].unique()

    video_path = "D:/DATASET/Kinetics/Video/"
    for label in labels:
        label = label.strip()
        if os.path.exists(video_path + label) == False:
            os.mkdir(video_path + label)
        else:
            continue

    for index, row in file.iterrows():

        # print(row)
        label, youtube_id = row['label'].strip(), row['youtube_id'].strip()

        # print(str(index)+", "+label, youtube_id)
        format_list = ['.webm', '.mp4']
        exist = False
        for file_format in format_list:
            if os.path.isfile(video_path + label + '/' + label + '_' + str(index) + file_format):
                exist = True
                # print(label + '_' + str(index) + " already exists!!")
                continue

        if not exist:
            try:
                print(youtube_id)
                yt = pytube.YouTube("https://www.youtube.com/watch?v="+youtube_id)
                yt.streams.filter(progressive=True).desc().first(). \
                    download(video_path + label + '/', filename_prefix=label + "_", filename=str(index))
                # print(label + "_" + str(index) + " 다운완료")
            except pytube.exceptions.VideoUnavailable:
                print(youtube_id + " 동영상 존재하지 않음")
                pass
    '''
    pool = multiprocessing.Pool(processes=6)
    x = pool.map(multi_thread, range(len(file)))
    x.get()
    pool.close()
    pool.join()
    '''