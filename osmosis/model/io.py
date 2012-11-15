"""
File handling for the model module. 

"""

def params_file_resolver(object, file_name_root, params_file=None):
    """
    Helper function for resolving what the params file name should be for
    several of the model functions for which the params are cached to file

    Parameters
    ----------
    object: the class instance affected by this

    file_name_root: str, the string which will typically be added to the
        file-name of the object's data file in generating the model params file. 

    params_file: str or None
       If a string is provided, this will be treated as the full path to where
       the params file will get saved. This will be defined if the user
       provides this as an input to the class constructor.

    Returns
    -------
    params_file: str, full path to where the params file will eventually be
            saved, once parameter fitting is done.
    
    """
    # If the user provided
    if params_file is not None: 
        return params_file
    else:
        # If our DWI super-object has a file-name, construct a file-name out of
        # that:
        if hasattr(object, 'data_file'):
            path, f = os.path.split(object.data_file)
            # Need to deal with the double-extension in '.nii.gz':
            file_parts = f.split('.')
            name = file_parts[0]
            extension = ''
            for x in file_parts[1:]:
                extension = extension + '.' + x
                params_file = os.path.join(path, name +
                                           file_name_root +
                    extension)
        else:
            # Otherwise give up and make a file right here with a generic
            # name: 
            params_file = '%s.nii.gz'%file_name_root

    return params_file
