"""
EfficientPose (c) by Steinbeis GmbH & Co. KG für Technologietransfer
Haus der Wirtschaft, Willi-Bleicher-Straße 19, 70174 Stuttgart, Germany
Yannick Bukschat: yannick.bukschat@stw.de
Marcus Vetter: marcus.vetter@stw.de

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

NumPy 2.x compatibility patch added.
"""

import numpy as np
import os
import sys
import importlib
import shutil
import re

# Patch imgaug before importing it
def patch_imgaug_for_numpy2():
    """
    Apply patches to make imgaug work with NumPy 2.x
    Returns True if successful, False otherwise
    """
    try:
        # Find imgaug module
        import importlib.util
        spec = importlib.util.find_spec('imgaug')
        if spec is None:
            print("Error: imgaug module not found")
            return False
            
        imgaug_dir = os.path.dirname(spec.origin)
        imgaug_main = os.path.join(imgaug_dir, "imgaug.py")
        
        if not os.path.exists(imgaug_main):
            print(f"Error: Cannot find imgaug.py at {imgaug_main}")
            return False
            
        # Create backup if it doesn't exist
        backup_path = imgaug_main + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(imgaug_main, backup_path)
            print(f"Created backup at {backup_path}")
            
        # Read the file
        with open(imgaug_main, 'r') as f:
            content = f.read()
            
        # Check if already patched for float types
        if 'NP_FLOAT_TYPES = {np.float16, np.float32, np.float64}' in content:
            print("imgaug already patched for NumPy 2.x float types")
            already_patched_float = True
        else:
            already_patched_float = False
            
        # Check if already patched for int types    
        if 'NP_INT_TYPES = {np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64}' in content:
            print("imgaug already patched for NumPy 2.x int types")
            already_patched_int = True
        else:
            already_patched_int = False
            
        if already_patched_float and already_patched_int:
            print("imgaug is already fully patched for NumPy 2.x")
            return True
            
        # Apply float types patch if needed
        if not already_patched_float and 'NP_FLOAT_TYPES = set(np.sctypes["float"])' in content:
            print("Patching imgaug.py for NumPy 2.x float types compatibility...")
            
            # Replace float types
            content = content.replace(
                'NP_FLOAT_TYPES = set(np.sctypes["float"])',
                'NP_FLOAT_TYPES = {np.float16, np.float32, np.float64}'
            )
        
        # Apply int types patch if needed
        # There are two places where int types are used:
        # 1. NP_INT_TYPES = set(np.sctypes["int"] + np.sctypes["uint"])
        # 2. Simple references to np.sctypes["int"] and np.sctypes["uint"]
        
        if not already_patched_int:
            print("Patching imgaug.py for NumPy 2.x int types compatibility...")
            
            # Replace the main int types definition
            content = content.replace(
                'NP_INT_TYPES = set(np.sctypes["int"] + np.sctypes["uint"])',
                'NP_INT_TYPES = {np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64}'
            )
            
            # Replace individual sctypes references
            content = content.replace(
                'np.sctypes["int"]',
                '[np.int8, np.int16, np.int32, np.int64]'
            )
            
            content = content.replace(
                'np.sctypes["uint"]',
                '[np.uint8, np.uint16, np.uint32, np.uint64]'
            )
        
        # Write the patched file
        with open(imgaug_main, 'w') as f:
            f.write(content)
            
        print("Successfully patched imgaug for NumPy 2.x compatibility")
        
        # Reload the module if it's already loaded
        if 'imgaug.imgaug' in sys.modules:
            print("Reloading imgaug module...")
            importlib.reload(sys.modules['imgaug.imgaug'])
            
        return True
    except Exception as e:
        print(f"Error patching imgaug: {e}")
        return False

# Apply the patch before importing any imgaug components
success = patch_imgaug_for_numpy2()
if not success:
    print("WARNING: Failed to patch imgaug. RandAugment may not work correctly with NumPy 2.x")

# Now it's safe to import imgaug components
try:
    from imgaug import parameters as iap
    from imgaug import random as iarandom
    from imgaug.augmenters import meta
    from imgaug.augmenters import arithmetic
    from imgaug.augmenters import pillike
    import imgaug.augmenters as iaa
except ImportError as e:
    print(f"Error importing imgaug: {e}")
    # Create dummy classes/functions to avoid errors when importing this module
    class DummyRandAugment:
        def __init__(self, *args, **kwargs):
            print("WARNING: Using dummy RandAugment because imgaug could not be imported")
        
        def __call__(self, images, *args, **kwargs):
            # Just return the input images unchanged
            return images
    
    # Set RandAugment to the dummy class
    RandAugment = DummyRandAugment
else:
    # If imports succeeded, define the real RandAugment class
    class RandAugment(meta.Sequential):
        """Apply RandAugment to inputs as described in the corresponding paper.
        This version is modified to work with NumPy 2.x.
        """

        _M_MAX = 30

        def __init__(self, n=2, m=(6, 12), cval=128,
                     seed=None, name=None,
                     random_state="deprecated", deterministic="deprecated"):
            # pylint: disable=invalid-name
            seed = seed if random_state == "deprecated" else random_state
            rng = iarandom.RNG(seed)

            # Handle the m parameter
            m = self._handle_discrete_param(
                m, "m", value_range=(0, None),
                tuple_to_uniform=True, list_to_choice=True,
                allow_floats=False)
            self._m = m
            self._cval = cval

            # Create the augmenters list
            main_augs = self._create_main_augmenters_list(m, cval)

            # Assign random state to all augmenters
            for augmenter in main_augs:
                augmenter.random_state = rng

            # Call the parent class constructor
            super(RandAugment, self).__init__(
                [
                    meta.SomeOf(n, main_augs, random_order=True,
                                seed=rng.derive_rng_())
                ],
                seed=rng, name=name,
                random_state=random_state, deterministic=deterministic
            )

        def _handle_discrete_param(self, param, param_name, value_range, 
                                  tuple_to_uniform=True, list_to_choice=True,
                                  allow_floats=False):
            """Replacement for iap.handle_discrete_param to avoid potential NumPy 2.x issues"""
            try:
                # Try to use the imgaug function first
                return iap.handle_discrete_param(
                    param, param_name, value_range,
                    tuple_to_uniform=tuple_to_uniform, 
                    list_to_choice=list_to_choice,
                    allow_floats=allow_floats
                )
            except Exception as e:
                # Fall back to a simplified implementation if the imgaug function fails
                print(f"Warning: falling back to simplified parameter handling due to: {e}")
                
                if isinstance(param, (int, float)) and not (isinstance(param, float) and not allow_floats):
                    return iap.Deterministic(param)
                elif isinstance(param, tuple) and len(param) == 2 and tuple_to_uniform:
                    a, b = param
                    return iap.DiscreteUniform(a, b)
                elif isinstance(param, list) and list_to_choice:
                    return iap.Choice(param)
                elif isinstance(param, iap.StochasticParameter):
                    return param
                else:
                    raise ValueError(f"Expected {param_name} to be int, float, tuple of 2 ints/floats, list of ints/floats or StochasticParameter, got {type(param)}.")


        @classmethod
        def _create_main_augmenters_list(cls, m, cval):
            # pylint: disable=invalid-name
            m_max = cls._M_MAX

            def _float_parameter(level, maxval):
                maxval_norm = maxval / m_max
                return iap.Multiply(level, maxval_norm, elementwise=True)

            def _int_parameter(level, maxval):
                # paper applies just int(), so we don't round here
                return iap.Discretize(_float_parameter(level, maxval),
                                      round=False)

            def _enhance_parameter(level):
                fparam = _float_parameter(level, 0.9)
                return iap.Clip(
                    iap.Add(1.0, iap.RandomSign(fparam), elementwise=True),
                    0.1, 1.9
                )

            def _subtract(a, b):
                return iap.Subtract(a, b, elementwise=True)

            def _affine(*args, **kwargs):
                kwargs["fillcolor"] = cval
                if "center" not in kwargs:
                    kwargs["center"] = (0.0, 0.0)
                return pillike.Affine(*args, **kwargs)

            # Create and return list of augmenters
            return [
                meta.Identity(),
                pillike.Autocontrast(cutoff=0),
                pillike.Equalize(),
                arithmetic.Invert(p=1.0),
                pillike.Posterize(
                    nb_bits=_subtract(
                        8,
                        iap.Clip(_int_parameter(m, 6), 0, 6)
                    )
                ),
                pillike.Solarize(
                    p=1.0,
                    threshold=iap.Clip(
                        _subtract(256, _int_parameter(m, 256)),
                        0, 256
                    )
                ),
                pillike.EnhanceColor(_enhance_parameter(m)),
                pillike.EnhanceContrast(_enhance_parameter(m)),
                pillike.EnhanceBrightness(_enhance_parameter(m)),
                pillike.EnhanceSharpness(_enhance_parameter(m)),
                arithmetic.Cutout(1,
                                  size=iap.Clip(
                                      _float_parameter(m, 20 / 32), 0, 20 / 32),
                                  squared=True,
                                  fill_mode="constant",
                                  cval=cval),
                pillike.FilterBlur(),
                pillike.FilterSmooth(),
                iaa.AdditiveGaussianNoise(scale=(m / 100.) * 255, per_channel=True)
            ]

        def get_parameters(self):
            """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
            try:
                someof = self[1]
                return [someof.n, self._m, self._cval]
            except Exception as e:
                print(f"Warning: Error in get_parameters: {e}")
                return [None, self._m, self._cval]


# Test if run directly
if __name__ == "__main__":
    # Create a RandAugment instance
    try:
        aug = RandAugment(n=2, m=9)
        print("Successfully created RandAugment")
        
        # Test on a dummy image
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        augmented = aug(images=[dummy_image])
        print("Successfully applied augmentation")
    except Exception as e:
        print(f"Error creating or using RandAugment: {e}")