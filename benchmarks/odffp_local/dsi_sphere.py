
import os.path

from dipy.core.sphere import Sphere
from scipy.io import loadmat

  
class _DsiSphere8Fold(Sphere):
    _instance = None

    @staticmethod 
    def get_instance():
        if _DsiSphere8Fold._instance == None:
            _DsiSphere8Fold()
        return _DsiSphere8Fold._instance
    
    
    def __init__(self):
        if _DsiSphere8Fold._instance != None:
            raise Exception("The class _DsiSphere8Fold is a singleton. Call dsiSphere8Fold() function instead.")
        else:
            dsi_sphere = loadmat(os.path.join(
                os.path.dirname(__file__), "dsi_sphere_8fold.mat"
            ))
            Sphere.__init__(self,
                xyz=dsi_sphere['odf_vertices'].T,
                faces=dsi_sphere['odf_faces'].T
            )
            _DsiSphere8Fold._instance = self


def dsiSphere8Fold():
    return _DsiSphere8Fold.get_instance()
        
        
    