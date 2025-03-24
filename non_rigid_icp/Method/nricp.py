import numpy as np
import open3d as o3d
from typing import Union
from scipy import sparse
from copy import deepcopy
from sksparse.cholmod import cholesky_AAt
from sklearn.neighbors import NearestNeighbors


def choleskySolve(M, b):
    factor = cholesky_AAt(M.T)
    return factor(M.T.dot(b)).toarray()

def pointsNonRigidICP(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    source_triangles: np.ndarray,
    source_normals: Union[np.ndarray, None]=None,
    target_normals: Union[np.ndarray, None]=None,
    normal_weighting: bool = False,
    gamma: float = 1.0,
    alphas: Union[list, np.ndarray] = np.linspace(200, 1, 20),
) -> np.ndarray:
    n_soutce_pts = source_pts.shape[0]

    knnsearch = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_pts)

    alledges=[]
    for face in source_triangles:
        face = np.sort(face)
        alledges.append(tuple([face[0],face[1]]))
        alledges.append(tuple([face[0],face[2]]))
        alledges.append(tuple([face[1],face[2]]))

    edges = set(alledges)
    n_source_edges = len(edges)

    M = sparse.lil_matrix((n_source_edges, n_soutce_pts), dtype=np.float32)

    for i, t in enumerate(edges):
        M[i, t[0]] = -1
        M[i, t[1]] = 1

    G = np.diag([1, 1, 1, gamma]).astype(np.float32)

    kron_M_G = sparse.kron(M, G)

    # X for transformations and D for vertex info in sparse matrix
    # using lil_matrix becaiuse chinging sparsity in csr is expensive
    #Equation -> 8
    D = sparse.lil_matrix((n_soutce_pts,n_soutce_pts*4), dtype=np.float32)
    j_=0
    for i in range(n_soutce_pts):
        D[i,j_:j_+3]=source_pts[i,:]
        D[i,j_+3]=1
        j_+=4

    #AFFINE transformations stored in the 4n*3 format
    X_= np.concatenate((np.eye(3),np.array([[0,0,0]])),axis=0)
    X = np.tile(X_,(n_soutce_pts,1))

    if normal_weighting:
        n_source_normals = len(source_normals) #will be equal to n_soutce_pts
        DN = sparse.lil_matrix((n_source_normals,n_source_normals*4), dtype=np.float32)
        j_=0
        for i in range(n_source_normals):
            DN[i,j_:j_+3]=source_normals[i,:]
            DN[i,j_+3]=1
            j_+=4

    for num_,alpha_stiffness in enumerate(alphas):

        print("step- {}/20".format(num_+1))

        for i in range(3):

            wVec = np.ones((n_soutce_pts,1))

            vertsTransformed = D*X

            distances, indices = knnsearch.kneighbors(vertsTransformed)

            indices = indices.squeeze()

            matches = target_pts[indices]

            #rigtnow setting threshold manualy, but if we have and landmark info we could set here
            mismatches = np.where(distances>0.02)[0]

            if normal_weighting:
                normalsTransformed = DN*X
                corNormalsTarget = target_normals[indices]
                crossNormals = np.cross(corNormalsTarget, normalsTransformed)
                crossNormalsNorm = np.sqrt(np.sum(crossNormals**2,1))
                dotNormals = np.sum(corNormalsTarget*normalsTransformed,1)
                angles =np.arctan(dotNormals/crossNormalsNorm)
                wVec = wVec *(angles<np.pi/4).reshape(-1,1)

            #setting weights of false mathces to zero   
            wVec[mismatches] = 0

            #Equation  12
            #E(X) = ||AX-B||^2

            U = wVec*matches

            A = sparse.csr_matrix(sparse.vstack([alpha_stiffness * kron_M_G, D.multiply(wVec)]))

            B = sparse.lil_matrix((4 * n_source_edges + n_soutce_pts, 3), dtype=np.float32)

            B[4 * n_source_edges: (4 * n_source_edges +n_soutce_pts), :] = U

            X = choleskySolve(A, B)

    vertsTransformed = D*X;

    #project source on to template
    matcheindices = np.where(wVec > 0)[0]
    vertsTransformed[matcheindices]=matches[matcheindices]

    return vertsTransformed

def nonrigidIcp(
    sourcemesh: o3d.geometry.TriangleMesh,
    targetmesh: o3d.geometry.TriangleMesh,
    normal_weighting: bool = False,
    gamma: float = 1.0,
    alphas: Union[list, np.ndarray] = np.linspace(200, 1, 20),
):
    refined_sourcemesh = deepcopy(sourcemesh)
    #obtain vertices
    target_vertices = np.array(targetmesh.vertices)
    source_vertices = np.array(refined_sourcemesh.vertices)
    sourcemesh_faces = np.array(sourcemesh.triangles)

    deformed_pts = pointsNonRigidICP(
        source_vertices,
        target_vertices,
        sourcemesh_faces,
        None,
        None,
        normal_weighting,
        gamma,
        alphas,
    )

    refined_sourcemesh.vertices = o3d.utility.Vector3dVector(deformed_pts)

    return refined_sourcemesh
