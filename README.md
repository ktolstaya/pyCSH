# PyCSH


Implementation of Coherency Sensitive Hashing algorithm

@author: Ekaterina Tolstaya

This program is based on the code of "Coherency Sensitive Hashing", Matlab and C++
by Simon Korman and Shai Avidan

http://www.eng.tau.ac.il/~simonk/CSH/

Coherency Sensitive Hashing (CSH) extends Locality Sensitivity Hashing (LSH) 
and PatchMatch to quickly find matching patches between two images. LSH relies 
on hashing, which maps similar patches to the same bin, in order to find matching 
patches. PatchMatch, on the other hand, relies on the observation that images are 
coherent, to propagate good matches to their neighbors, in the image plane. It 
uses random patch assignment to seed the initial matching. CSH relies on hashing 
to seed the initial patch matching and on image coherence to propagate good matches. 
In addition, hashing lets it propagate information between patches with similar 
appearance (i.e., map to the same bin). This way, information is propagated much 
faster because it can use similarity in appearance space or neighborhood in the 
image plane. As a result, CSH is at least three to four times faster than PatchMatch 
and more accurate, especially in textured regions, where reconstruction artifacts 
are most noticeable to the human eye.

--------------------------------------------

The CSH algorithm was applied in my depth propagation project:

https://www.researchgate.net/publication/282681757_Depth_propagation_for_semi-automatic_2D_to_3D_conversion

