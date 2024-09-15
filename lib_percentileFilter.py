# coding: utf-8
'''
異方性パーセンタイルフィルタをかける  
使用例 (使い方) は最下部
'''

import numpy as np
from numba import jit

# --------------------------------------------------
# median filters
# --------------------------------------------------

@jit('float32[:,:](float32[:,:], float32[:,:], int16[:], int16[:], int16[:], int16[:], float32[:])', nopython=True, cache=True)
def _calcPercentileMat_Gray(
    srcMat: np.ndarray,
    tgtMat: np.ndarray,
    startYs: np.ndarray,
    stopYs: np.ndarray,
    startXs: np.ndarray,
    stopXs: np.ndarray,
    percentile: np.ndarray,
):
    for y in range(srcMat.shape[0]):
        for x in range(srcMat.shape[1]):
            if not(np.isnan(srcMat[y, x])):
                tgtMat[y, x] = np.nanpercentile(
                    srcMat[startYs[y]: stopYs[y], startXs[x]: stopXs[x]],
                    q=percentile[0]
                )
    return tgtMat


def percentileFilter(
    srcMat: np.ndarray,
    ky: int,
    kx: int,
    foreMask: np.ndarray=None,
    percentile:float=50.,
):
    '''
    パーセンタイル画像を計算する．  

    [引数]
    - srcMat (np.ndarray): 画像 (整数型でも小数型でもグレースケールでも bgr でも可)
    - ky: y 方向の kernel size
    - kx: x 方向の kernel size
    - foreMask (np.ndarray, 全ての要素は 0 または 1 を取る)=None : 正規積分のための前景マスク (整数型でも小数型でも可．入力がない場合は画像全面を前景と見做す)
    - percentile (float)=50. : パーセンタイル値．50.0 を入れると median filter として機能する．

    [戻り値]
    計算結果画像 (srcMat と同じ dtype)
    '''
    
    if len(srcMat.shape) == 2:
        isGrayscale = True
    elif len(srcMat.shape) == 3:
        assert srcMat.shape[2] == 3
        isGrayscale = False
    else:
        assert False


    matDtype = srcMat.dtype

    if foreMask is None:
        foreMask = np.ones([srcMat.shape[0], srcMat.shape[1]], dtype=np.float32)

    assert 0. <= percentile <= 100.
    assert np.max(foreMask) < 1.0001
    assert np.min(foreMask) > -0.0001

    startYs = np.linspace(0, srcMat.shape[0]-1, srcMat.shape[0]) - ky//2
    startXs = np.linspace(0, srcMat.shape[1]-1, srcMat.shape[1]) - kx//2
    stopYs = startYs + ky
    stopXs = startXs + kx

    startYs = np.clip(startYs, 0, None).astype(np.int16)
    startXs = np.clip(startXs, 0, None).astype(np.int16)
    stopYs = stopYs.astype(np.int16)
    stopXs = stopXs.astype(np.int16)

    # 正規計算用マスクの処理
    _srcMat = srcMat.astype(np.float32)
    tgtMat = np.zeros_like(_srcMat)

    if foreMask is not None:
        if isGrayscale:
            _srcMat[foreMask < 0.5] = np.nan
            tgtMat[foreMask < 0.5] = np.nan
        else:
            for c in [0, 1, 2]:
                _srcMat[foreMask < 0.5, c] = np.nan
                tgtMat[foreMask < 0.5, c] = np.nan

    if isGrayscale:
        tgtMat = _calcPercentileMat_Gray(
            srcMat=_srcMat,
            tgtMat=tgtMat,
            startYs=startYs,
            stopYs=stopYs,
            startXs=startXs,
            stopXs=stopXs,
            percentile=np.array([percentile,], dtype=np.float32)
        )

    else:
        for c in [0, 1, 2]:
            tgtMat[:, :, c] = _calcPercentileMat_Gray(
                srcMat=_srcMat[:, :, c],
                tgtMat=tgtMat[:, :, c],
                startYs=startYs,
                stopYs=stopYs,
                startXs=startXs,
                stopXs=stopXs,
                percentile=np.array([percentile,], dtype=np.float32)
            )
    
    if foreMask is not None:
        tgtMat = np.nan_to_num(tgtMat)
    return tgtMat.astype(matDtype)


if __name__ == "__main__":
    '''
    使用例
    '''

    import matplotlib.pyplot as plt
    import cv2
    import time # 計算時間計測用

    # -------------------------------------------------------
    # 画像の読み込み
    # -------------------------------------------------------

    srcMat = cv2.imread(r"sampleData\0003.png", cv2.IMREAD_GRAYSCALE)

    # 正規パーセンタイル計算を行いたい場合は foreMask も入れる．特に正規な計算を求めないなら不要．
    foreMask = cv2.imread(r"sampleData\0003_foremask.png", cv2.IMREAD_GRAYSCALE)

    t0 = time.time() # 計算時間計測用

    # -------------------------------------------------------
    # パーセンタイルフィルタ
    # -------------------------------------------------------
    
    medianMat = percentileFilter(
        srcMat,
        ky=10,
        kx=100,
        foreMask=foreMask, # optinal 引数．何も入れない場合は全体を前景と見做す．
        percentile=50., # optional 引数．何も入れない場合は 50% つまり中央値フィルタとなる．
    )

    # -------------------------------------------------------
    # 結果表示
    # -------------------------------------------------------

    print('計算時間:', time.time() - t0) # 計算時間計測用
    plt.figure(figsize=(12,6))
    ax = plt.subplot(1,2,1)
    im = ax.imshow(srcMat, cmap='jet', vmin=0, vmax=191)
    plt.colorbar(im)
    plt.title('original mat')
    ax = plt.subplot(1,2,2)
    im = ax.imshow(medianMat, cmap='jet', vmin=0, vmax=191)
    plt.colorbar(im)
    plt.title('median mat')
    plt.show()