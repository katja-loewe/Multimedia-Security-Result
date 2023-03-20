import numpy as np, cv2, glob, time, itertools, scipy
from operator import itemgetter

# Predefined Parameters:
# bsize is blocksize = 8
# variables with T could be Thresholds
tf, qf, Na, St, Tt, Ct, Tf, Td, bsize = 4, .25, 3, 3, 0.06, 3, 50, 40, 8

# zz_mat is a matrix that is used to organise a block as a zig zag pattern
zz_mat = [0,0, 0,1, 1,0, 2,0, 1,1, 0,2, 0,3, 1,2, 2,1, 3,0, 4,0, 3,1, 2,2, 1,3, 0,4, 0,5, 1,4, 2,3, 3,2, 4,1, 5,0, 6,0, 5,1, 4,2, 3,3, 2,4, 1,5, 0,6, 0,7, 1,6, 2,5, 3,4, 4,3, 5,2, 6,1, 7,0]
# zzidx is not used in the program
zzidx = [0,0, 0,1, 1,0, 2,0, 1,1, 0,2, 0,3, 1,2, 2,1, 3,0, 3,1, 2,2, 1,3, 2,3, 3,2, 3,3]


# DCT and Zigzag-Sort on disjoint blocks of size B
def dct_blocks(img, B):
    h, w = img.shape
    img = img.astype(float)

    # blocks is a 4D numpy array which contains blocks of size 8x8 each
    # the blocks are shifted by 1 value, this means we have (h - B + 1) * (w - B + 1) times an 8x8 block
    blocks = np.lib.stride_tricks.as_strided(img, shape=(h - B + 1, w - B +1, B, B), strides=img.strides + img.strides)
    bh, bw = blocks.shape[:2]

    # for each block in blocks, a dct is performed
    # next, each value is multiplied by 0.25, then added by 0.5, and then rounded to an integer - still in block shape
    # next, a zig zag algorithm is used to store the block in a list
    # within in zigzag_sort, only the first 16 values out of 64 are kept, the rest is discarded
    dct = [[(i, j), zigzag_sort((cv2.dct(blocks[i][j]) * qf + .5).astype(int).tolist())] for (i,j) in itertools.product(range(bh), range(bw))]
    return dct

# sorts a block as a list, by making use of a zigzag pattern
# the corresponding elements are stored in zz_mat in the beginning of the program
# only 2*tf**2 values are used; tf = 4 which makes a range from 0 to 32 with a step of 2
# -> the first 16 values are stored in b
# -> extracts only the main frequencies but with a hard coded threshold
def zigzag_sort(a):
    b = []
    for i in range(0, 2 * tf**2, 2):
        b.append(a[zz_mat[i]][zz_mat[i + 1]])
    return b

# the dictionary is sorted in descending order by the length of the values list in the dictionary tv
# only the first 4 rows are kept
# all indices that are in the length of tv are marked white in the result image
def get_image_results(img, tv):
    img_res = np.zeros(img.shape, dtype = 'uint8')
    s = sorted(tv, key=lambda k: len(tv[k]), reverse=True)[:4]

    for k in s:
        print(f"k: ", k)
        v = tv[k]
        # Tf = 50
        if len(v) > Tf:
            for x in v:
                img_res[x[0], x[1]] = 1
                img_res[x[0] + k[0], x[1] + k[1]] = 1
    return img_res

# A consists of 3 lists:
# -> they are next to each other in a sorted version of dct's, which means their low frequency dct coefficients are similar
# tv = transfer vector is an empty dictionary
# the function compares the second list with the first list, and then the third list with the first list
def compare(A, tv):

    # checks if A is empty
    if A == None or len(A) == 0:
        return

    # stores the first list in a
    a = A[0]

    # b is the second list, later b is the third list
    for b in A[1:]:

        # repeating the steps before ... stores the lists in ai and aj
        ai, aj = np.array(a[1]), np.array(b[1])

        # the lists corresponding indexes (pointing to the block location) are stored here
        idxi, idxj = np.array(a[0]), np.array(b[0])

        # v is the difference of the two lists indices
        v = (idxj[0] - idxi[0], idxj[1] - idxi[1])

        # if the norm of v > Td = 40 (predefined) -> then the blocks of the two successive lists are far away in the img
        # here the copy move detection is done -> if the indices are distant (>Td), it's possibly a copy move detection!
        if np.linalg.norm(v) > Td:

            rmax, rmin = -5000, 5000
            c = 0

            # iterates over the all elements of the second and third lists
            # if the element == 0 and absolut value of the difference of the iterated item of both lists ...
            # ... is >= St = 3 (predefined), increment the counter
            # else the iterated items are divided, compared with a set r and replace possibly r
            for l in range(len(aj)):
                if aj[l] == 0:
                    # if the absolute difference between the elements is >= St = 3, the locations are possibly not ...
                    # ... next to each other -> counter is incremented
                    if abs(ai[l] - aj[l]) >= St:
                        c += 1
                else:
                    r = 1.0*ai[l] / aj[l]
                    if r < rmin:
                        rmin = r
                    if r > rmax:
                        rmax = r

            # Also if rmax - rmin >= Tt = 0.06 (predefinded), the counter is incremented
            if rmax - rmin >= Tt:
                c += 1

            # if the counter is < Ct = 3 (predefinded),
            if c < Ct:
                idx = idxi
                if v[0] < 0:
                    v = (-v[0], -v[1])
                    idx = idxj
                if tv.get(v) == None:
                    tv[v] = []
                tv[v].append(idx)


def locate_copy_move_region_dct(img):
    # DCT and main frequency extraction: divides image into overlapping blocks, performs dct, divides values by 4, ...
    # ... adds 0.5, converts to int, uses the zigzag pattern to store the first 16 values out of 64
    # A is a list containing the location of each block and then an array of its 16 values
    start = time.time()
    A = dct_blocks(img, bsize)
    print('DCT: {} seconds'.format(time.time() - start))

    # A is sorted in a descending order by the 1st value out of the 16; the location of each block is still a prior note
    start = time.time()
    A = sorted(A, key=itemgetter(1), reverse=True)
    print('SORT: {} seconds'.format(time.time() - start))

    # tv is Dictionary of transfer vectors (armas original comment)
    # AA is a list of overlapping sublists which have the length of Na = 3 (predefined in the beginning)
    start = time.time()
    tv = {}
    AA = list(map(lambda i: A[i: i+Na], list(range(len(A) - Na))))
    print('NEIGH: {} seconds'.format(time.time() - start))

    # the function compare takes each sublist of AA and the empty transfer vector
    # in the function compare, the tv dictionary is expanded every time when the
    start = time.time()
    list(map(lambda X: compare(X, tv), AA))
    print('COMPARE: {} seconds'.format(time.time() - start))

    # dictionary is sorted in descending order and only indices in first 4 rows are then marked in white as forged
    start = time.time()
    img_res = get_image_results(img, tv)
    print('RESULTS: {} seconds'.format(time.time() - start))

    return img_res


# read image in grayscale
path_in = 'D:/semester01/Multimedia-Security/armas-dct-copy/images/giraffe_copy.png'
img = cv2.imread(path_in, 0)

# perform armas-dct-program
result = locate_copy_move_region_dct(img)

# save image
path_out = 'D:/semester01/Multimedia-Security/armas-dct-copy/images/giraffe_copy-result.png'
cv2.imwrite(path_out, result*255)