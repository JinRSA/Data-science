import os
from os.path import join, exists
# import multiprocessing
import multiprocessing.dummy# as MPD
import hashlib
import cv2
from tqdm import tqdm
import numpy as np

files = ['facescrub_actors.txt', 'facescrub_actresses.txt']
files = ['facescrub_actresses.txt']
allFilesCount, failedToDownloadCounter = 0, 0
RESULT_ROOT = "Images"
if not exists(RESULT_ROOT):
    os.mkdir(RESULT_ROOT)

def download(names, urls, bboxes, gender):
    """
        Download from urls into folder names using wget.
    """
    assert(len(names) == len(urls))
    assert(len(names) == len(bboxes))

    failedToDownloadCounter = 0
    # Download using external wget.
    CMD = 'wget -c -t 1 -T 3 "%s" -O "%s"'
    for i in range(len(names)):
        directory = join(RESULT_ROOT, names[i])
        if not exists(directory):
            os.mkdir(directory)
        fname = gender[i] + "_" + hashlib.sha1(urls[i].encode("utf-8")).hexdigest() + '.jpg'
        dst = join(directory, fname)
        if exists(dst):
            print("Already downloaded, skipping...")
            continue
        else:
            res = os.system(CMD % (urls[i], dst))
        # Get face
        face_directory = join(directory, 'face')
        if not exists(face_directory):
            os.mkdir(face_directory)
        img = cv2.imread(dst)
        if img is None:
            # No image data.
            os.remove(dst)
            print(f"Failed to download. URL: {urls[i]}")
            failedToDownloadCounter += 1
        else:
            face_path = join(face_directory, fname)
            face = img[bboxes[i][1]:bboxes[i][3], bboxes[i][0]:bboxes[i][2]]
            cv2.imwrite(face_path, face)
            # Write bbox to file.
            with open(join(directory,'_bboxes.txt'), 'a') as fd:
                bbox_str = ','.join([str(_) for _ in bboxes[i]])
                fd.write('%s %s\n' % (fname, bbox_str))
    return failedToDownloadCounter

%%time
if __name__ == '__main__':
    print("Let's go!")
    for f in files:
        with open(f, 'r') as fd:
            # Strip first line.
            fd.readline()
            names = []
            urls = []
            bboxes = []
            gender = []
            thisGender = "M" if f == "facescrub_actors.txt" else "F"
            for line in fd.readlines():
                components = line.split('\t')
                assert(len(components) == 6)
                name = components[0].replace(' ', '_')
                url = components[3]
                bbox = [int(_) for _ in components[4].split(',')]
                names.append(name)
                urls.append(url)
                bboxes.append(bbox)
                gender.append(thisGender)
        # Every name gets a task.
        last_name = names[0]
        task_names = []
        task_urls = []
        task_bboxes = []
        task_gender = []
        tasks = []
        for i in range(len(names)):
            if names[i] == last_name:
                task_names.append(names[i])
                task_urls.append(urls[i])
                task_bboxes.append(bboxes[i])
                task_gender.append(gender[i])
            else:
                tasks.append((task_names, task_urls, task_bboxes, task_gender))
                task_names = [names[i]]
                task_urls = [urls[i]]
                task_bboxes = [bboxes[i]]
                task_gender = [gender[i]]
                last_name = names[i]
        tasks.append((task_names, task_urls, task_bboxes, task_gender))

        pool_size = multiprocessing.cpu_count()
        pool = multiprocessing.dummy.Pool(processes = pool_size * 4)
        Res = pool.starmap(download, tasks)
        pool.close()
        failedToDownloadCounter += np.sum(Res)
        allFilesCount += np.shape(names)[0]
        pool.join()
        print("1 file was processed.")
    print(f"Downloaded {allFilesCount - failedToDownloadCounter} files from {allFilesCount}.")
