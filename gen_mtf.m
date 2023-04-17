mtf_wv3 = mtf(0.25,'WV3',8);
mtf_wv3 = real(mtf_wv3);
isreal(mtf_wv3)
save('mtf_wv3.mat','mtf_wv3');
