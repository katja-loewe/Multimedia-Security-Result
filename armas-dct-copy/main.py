import numpy as np, cv2, glob, time, itertools, scipy
from operator import itemgetter

# Parameters
tf, qf, Na, St, Tt, Ct, Tf, Td, bsize = 4, .25, 3, 3, 0.06, 3, 50, 40, 8
zz_mat = [0,0, 0,1, 1,0, 2,0, 1,1, 0,2, 0,3, 1,2, 2,1, 3,0, 4,0, 3,1, 2,2, 1,3, 0,4, 0,5, 1,4, 2,3, 3,2, 4,1, 5,0, 6,0, 5,1, 4,2, 3,3, 2,4, 1,5, 0,6, 0,7, 1,6, 2,5, 3,4, 4,3, 5,2, 6,1, 7,0]
zzidx = [0,0, 0,1, 1,0, 2,0, 1,1, 0,2, 0,3, 1,2, 2,1, 3,0, 3,1, 2,2, 1,3, 2,3, 3,2, 3,3]


# DCT on disjoint blocks of size B
def dct_blocks(img, B):
    h, w = img.shape
    img = img.astype(float)
    blocks = np.lib.stride_tricks.as_strided(img, shape=(h - B + 1, w - B +1, B, B), strides=img.strides + img.strides)
    bh, bw = blocks.shape[:2]
    dct = [[(i, j), zigzag_sort((cv2.dct(blocks[i][j]) * qf + .5).astype(int).tolist())] for (i,j) in itertools.product(range(bh), range(bw))]
    return dct


def zigzag_sort(a):
    b = []
    for i in range(0, 2 * tf**2, 2):
        b.append(a[zz_mat[i]][zz_mat[i + 1]])
    return b


def get_image_results(img, tv):
    img_res = np.zeros(img.shape, dtype = 'uint8')
    s = sorted(tv, key = lambda k: len(tv[k]), reverse=True)[:4]
    for k in s:
        v = tv[k]
        if len(v) > Tf:
            for x in v:
                img_res[x[0], x[1]] = 1
                img_res[x[0] + k[0], x[1] + k[1]] = 1
    return img_res


def compare(A, tv):
    if A == None or len(A) == 0:
        return
    a = A[0]
    for b in A[1:]:
        ai, aj = np.array(a[1]), np.array(b[1])
        idxi, idxj = np.array(a[0]), np.array(b[0])
        v = (idxj[0] - idxi[0], idxj[1] - idxi[1])
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


def locate_copy_move_region_dct(img):
    # DCT and main frequency extraction
    start = time.time()
    A = dct_blocks(img, bsize)
    print('DCT: {} seconds'.format(time.time() - start))
    start = time.time()
    A = sorted(A, key=itemgetter(1), reverse=True)
    print('SORT: {} seconds'.format(time.time() - start))
    start = time.time()
    # Dictionary of transfer vectors
    tv = {}
    AA = list(map(lambda i: A[i: i+Na], list(range(len(A) - Na))))
    print('NEIGH: {} seconds'.format(time.time() - start))
    start = time.time()
    list(map(lambda X: compare(X, tv), AA))
    print('COMPARE: {} seconds'.format(time.time() - start))
    start = time.time()
    img_res = get_image_results(img, tv)
    print('RESULTS: {} seconds'.format(time.time() - start))
    return img_res




compression_range = [1, 2, 3, 4, 5, 6, 7, 8, 9]



# test_img = 'images/Au_ani_00024.jpg'
# img = cv2.imread(test_img, 0)
# result = locate_copy_move_region_dct(img)
# path = 'images/Au_ani_00024_jpg.jpg'
# cv2.imwrite(path, result*255)

for image in range(1,10):
    print(image)

    for compression in compression_range:
        print(compression)

        test_img = 'images/00' + str(image) + '_F_JC' + str(compression) + '.jpg'
        #test_img = 'images/003_F_JC8.jpg'
        img = cv2.imread(test_img, 0)
        result = locate_copy_move_region_dct(img)
        path = 'images/00' + str(image) + '_F_JC' + str(compression) + '.jpg-result.jpg'
        cv2.imwrite(path, result*255)

for image in range(10,20):
    print(image)

    for compression in compression_range:
        print(compression)

        test_img = 'images/0' + str(image) + '_F_JC' + str(compression) + '.jpg'
        #test_img = 'images/003_F_JC8.jpg'
        img = cv2.imread(test_img, 0)
        result = locate_copy_move_region_dct(img)
        path = 'images/0' + str(image) + '_F_JC' + str(compression) + '.jpg-result.jpg'
        cv2.imwrite(path, result*255)

for image in range(20, 21):
    print(image)

    for compression in compression_range:
        print(compression)

        test_img = 'images/0' + str(image) + '_F_JC' + str(compression) + '.jpg'
        #test_img = 'images/003_F_JC8.jpg'
        img = cv2.imread(test_img, 0)
        result = locate_copy_move_region_dct(img)
        path = 'images/0' + str(image) + '_F_JC' + str(compression) + '.jpg-result.jpg'
        cv2.imwrite(path, result*255)