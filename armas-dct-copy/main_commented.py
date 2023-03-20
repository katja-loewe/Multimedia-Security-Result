import numpy as np, cv2, glob, time, itertools, scipy
from operator import itemgetter

# Parameters
tf, qf, Na, St, Tt, Ct, Tf, Td, bsize = 4, .25, 3, 3, 0.06, 3, 50, 40, 8
# transform
# tf = 4

# qf = .25
# Na = 3
# St = 3
# Tt = 0.06

# cosine transform?
# Ct = 3
# Tf = 50

# Threshold DCT ?
# Td = 40

# block size
# bsize = 8

# zz_mat: matrix for the zigzag sort, which element is used next
zz_mat = [0,0, 0,1, 1,0, 2,0, 1,1, 0,2, 0,3, 1,2, 2,1, 3,0, 4,0, 3,1, 2,2, 1,3, 0,4, 0,5, 1,4, 2,3, 3,2, 4,1, 5,0, 6,0, 5,1, 4,2, 3,3, 2,4, 1,5, 0,6, 0,7, 1,6, 2,5, 3,4, 4,3, 5,2, 6,1, 7,0]
zzidx = [0,0, 0,1, 1,0, 2,0, 1,1, 0,2, 0,3, 1,2, 2,1, 3,0, 3,1, 2,2, 1,3, 2,3, 3,2, 3,3]


# DCT on disjoint blocks of size B
def dct_blocks(img, B):

    #print(f"img: ", img)

    # get height and width of original image: h = 533, w = 800
    h, w = img.shape

    # convert img values to datatype float
    img = img.astype(float)

    # creates overlapping blocks
    # shape = (9, 9, 8, 8)
    # kreiert blocks der Größe 8x8, die aber jeweils um den Wert 1 überlappen
    # blocks sind numpy array
    blocks = np.lib.stride_tricks.as_strided(img, shape=(h - B + 1, w - B +1, B, B), strides=img.strides + img.strides)
    print(type(blocks))
    print("blocks")

    # img.strides = 6400, 8
    print(img.strides)

    # img.strides + img.strides = 6400, 8, 6400, 8
    print(img.strides + img.strides)

    #
    bh, bw = blocks.shape[:2]
    print(bh, bw)

    # blocks.shape = (526, 793, 8, 8)
    print(blocks.shape)

    # for every block a cv2.dct is performed
    # all elements are multiplied by 0.25 and added 0.5
    # they are then sorted by a zigzag pattern which is usual for dct
    # and then stored and returned by variable dct

    # is divided by 4: end result is 16 instead of 64 ...? CHECK THIS
    dct = [[(i, j), zigzag_sort((cv2.dct(blocks[i][j]) * qf + .5).astype(int).tolist())] for (i,j) in itertools.product(range(bh), range(bw))]
    print(type(dct[0]))
    print(len(dct))
    print((dct[0]))

    # dct is a list with length 417.118 = 526 * 793 = bh * bw
    # it contains first the block location and then the zigzag_sort
    return dct


# does the 1D zigzag_sort of a 2D block and TAKES ONLY THE FIRST 16 VALUES (incl. (0,5))
# a is 8x8 = 64 and b is 1x16 = 16
def zigzag_sort(a):
    b = []
    for i in range(0, 2 * tf**2, 2):
        b.append(a[zz_mat[i]][zz_mat[i + 1]])
    return b


def get_image_results(img, tv):
    # create empty = black result image
    img_res = np.zeros(img.shape, dtype = 'uint8')
    # sort in dictionary tv in descending oder
    s_strich = sorted(tv, key = lambda k: len(tv[k]), reverse=True)
    # sortiere die Werte absteigend und nimm nur die ersten 4 -> es sind 4x2 Tupl, evtl wieder Block indices
    s = sorted(tv, key = lambda k: len(tv[k]), reverse=True)[:4]
    print(f"hier: ", len(s_strich))
    #print(s)
    for k in s:
        print(k)
        v = tv[k]
        # Tf = 50
        #print(f"v: ", v)
        print(f"len(v): ", len(v))
        # len(v) gibt Anzahl der Werte in v
        # wenn len(v) > 50
        if len(v) > Tf:
            #wird für jedes tuple/block location in v die area auf 1 gesetzt -> forgery detected
            for x in v:
                img_res[x[0], x[1]] = 1
                #
                img_res[x[0] + k[0], x[1] + k[1]] = 1
    return img_res



def compare(A, tv):
    #print(f"A: ", A)
    #print(f"tv: ", tv)
    if A == None or len(A) == 0:
        return
    a = A[0]
    for b in A[1:]:
        #print(f"b: ", b)
        #print(f"hier: ", A[1:])
        ai, aj = np.array(a[1]), np.array(b[1])
        idxi, idxj = np.array(a[0]), np.array(b[0])
        v = (idxj[0] - idxi[0], idxj[1] - idxi[1])
        #print(f"a: ", a)
        #print(ai, aj, idxi, idxj, v)

        # this is in my pictures never the case, so this code snippet can be skipped
        if np.linalg.norm(v) > Td:
            rmax, rmin = -5000, 5000
            c = 0
            for l in range(len(aj)):
                if aj[l] == 0:
                    if abs(ai[l] - aj[l]) >= St:
                        c += 1
                else:
                    r = 1.0*ai[l] / aj[l]
                    if r < rmin:
                        rmin = r
                    if r > rmax:
                        rmax = r
            if rmax - rmin >= Tt:
                c += 1
            if c < Ct:
                idx = idxi
                if v[0] < 0:
                    v = (-v[0], -v[1])
                    idx = idxj
                if tv.get(v) == None:
                    tv[v] = []
                tv[v].append(idx)

        #print(f"tv: ", tv)


# DCT and main frequency extraction
def locate_copy_move_region_dct(img):

    # start timer
    start = time.time()

    # function divides img into dct-performable blocks, with default blocksize = 8, but in return each part is 16
    # elements long
    # A stores the DCT 1D list with 417.118 elements
    A = dct_blocks(img, bsize)
    #print(A)
    print('DCT: {} seconds'.format(time.time() - start))

    # start timer again
    start = time.time()

    # function sorts 1D blocks in A in a descending order (value in total of the block?)
    A = sorted(A, key=itemgetter(1), reverse=True)
    #print(A)
    print('SORT: {} seconds'.format(time.time() - start))

    # start timer again
    start = time.time()

    # Create empty Dictionary of transfer vectors
    tv = {}

    # It creates a list from 0 to len(A) - Na - 1, which will include all possible sublists of length Na = 3
    # Na = 3
    AA = list(map(lambda i: A[i: i+Na], list(range(len(A) - Na))))
    #print(f"AA: ", AA)
    print(len(AA))
    print('NEIGH: {} seconds'.format(time.time() - start))

    # start timer again
    start = time.time()

    # applies each element X of the list AA to the compare function, together with an empty dict tv
    list(map(lambda X: compare(X, tv), AA))
    #print(f"tv: ", tv)
    #print(f"2. AA: ", AA)
    print(len(AA))
    print('COMPARE: {} seconds'.format(time.time() - start))

    # start timer again
    start = time.time()

    # original image and dict of transfer vectors are given to calculate image results which are stores in img_res
    img_res = get_image_results(img, tv)
    print('RESULTS: {} seconds'.format(time.time() - start))

    # give back the image with found forgeries
    return img_res



# main:
# read image in grayscale from path and store it in img variable
test_img = 'images/cfa_0.png'
img = cv2.imread(test_img, 0)


# perform copy move detection
result = locate_copy_move_region_dct(img)

# write result image to path
cv2.imwrite('images/cfa_0.png-result.png', result*255)