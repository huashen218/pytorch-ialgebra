from ialgebra.utils.utils_interpreter import *


class Operator(object):
    """
    *Function*: 
    operate saliency_maps to meet user demands

    *Inputs*:
    :input:
    :model:
    :int_map: saliency_map generated by identity
    :interpreter_params:  set default if no user inputs
    :operator_params:

    *Returns*:
    :opt_map: shape = [B*C*H*W], type = numpy.ndarray
    :opt_map+img (might not use): shape = [B*C*H*W], type = numpy.ndarray 
    """
    def __init__(self, inputs_tup=None,  models_tup=None, identity_name=None, identity_kwargs=None, operators_tup=None, operators_kwargs_tup=None):
        self.inputs_tup = inputs_tup
        # parsing inputs
        self.bx, self.by = input_tup[0][0], input_tup[0][1] # input_tup = ((bx1, by1), (bx2,by2),...,(bxn,byn)), only use the 1st input 

        # parsing models
        self.model_kwargs = models_tup[0]                 # (model1_kwargs, model2_kwargs, ..., modeln_kwargs)
        # self.model = load_pretrained_model(**self.model_kwargs)

        # parsing identity
        self.identity = identity_name
        self.identity_kwargs = identity_kwargs

        # parsing operator
        self.operator = operators_tup[0]                  # (operator1, operator2 ,..., operatorn)
        self.operators_kwargs = operators_kwargs_tup[0]   # (operator1_kwargs, operator2_kwargs ,..., operatorn_kwargs)

        self.identity_interpreter = generate_identity(self.bx, self.by, self.model_kwargs, self.identity, self.identity_kwargs)



    def projection(self):
        heatmap, heatmapimg = self.identity_interpreter()
        return heatmap, heatmapimg



    def selection(self, region):
        # new input
        pos0, pos1, pos2, pos3 = region[0], region[1], region[2], region[3]
        img = self.bx
        mat = np.zeros(img.shape)
        roi = img[:, int(pos0):int(pos1), int(pos2):int(pos3)]
        mat[:, int(pos0):int(pos1), int(pos2):int(pos3)] = roi
        mat = mat.astype('float32')
                                                    ## wheter mat or bx??
        self.identity_interpreter = generate_identity(mat, self.by, self.model_kwargs, self.identity, self.identity_kwargs)
        heatmap, heatmapimg = self.identity_interpreter(mat)
        return heatmap, heatmapimg 



    # same_class x1, x2, model f1, 
    def join(self):
        """
        *Function*: 
        operater join: compare two inputs x and x' from same class and find most informative common features

        *Inputs*:
        :2 inputs: x, x'
        :1 model: f

        *Returns*:
        :common opt_map: 
        :opt_map+img_x:  
        :opt_map+img_x': 
        """
        bx1, by1 = self.input_tup[0][0], self.input_tup[0][1] 
        bx2, by2 = self.input_tup[1][0], self.input_tup[1][1]

        heatmap1, heatmapimg1 = self.identity_interpreter(bx1, by1)
        heatmap2, heatmapimg2 = self.identity_interpreter(bx2, by2)

        heatmap = 0.5 * (heatmap1 + heatmap2)
        
        heatmapimg1 =  heatmap + np.float32(bx1)
        heatmapimg1 = heatmapimg1 / np.max(heatmapimg1)

        heatmapimg2 =  heatmap + np.float32(bx2)
        heatmapimg2 = heatmapimg2 / np.max(heatmapimg2)

        return heatmap, heatmapimg1, heatmapimg2




    def antijoin(self, model_diff= False):
        """
        *Function*: 
        1: operater anti-join: compare two inputs x and x' from different classes and find most informative and discriminative features
        2: operater anti-join: compare one input x, and two different models f1, f2 with different classes, to find most informative and discriminative features

        *Inputs*:
        :2 inputs: x, x'
        :1 model: f

        *Returns*:
        :heatmap1
        :heatmapimg1
        :heatmap2
        :heatmapimg2
        """

        # case1: 1 input, 2 models
        if model_diff:
            bx = self.input_tup[0][0]
            self.model_kwargs1 = models_tup[0] 
            self.model1 = load_pretrained_model(**self.model_kwargs1)
            by1 = self.model1(bx)
            self.identity_interpreter1 = generate_identity(bx, by1, self.model_kwargs1, self.identity, self.identity_kwargs)


            self.model_kwargs2 = models_tup[1] 
            self.model2 = load_pretrained_model(**self.model_kwargs2)
            by2 = self.model2(bx)
            self.identity_interpreter2 = generate_identity(bx, by2, self.model_kwargs2, self.identity, self.identity_kwargs)

            heatmap1_1, heatmapimg1_1 = self.identity_interpreter1(bx, by1)  # interpreter1_cls1
            heatmap1_2, heatmapimg1_2 = self.identity_interpreter2(bx, by1)  # interpreter2_cls1
            heatmap2_1, heatmapimg2_1 = self.identity_interpreter1(bx, by2)  # interpreter1_cls2
            heatmap2_2, heatmapimg2_2 = self.identity_interpreter2(bx, by2)  # interpreter2_cls2

        # case2: 2 inputs, 1 model
        else:
            bx1, by1 = self.input_tup[0][0], self.input_tup[0][1] 
            bx2, by2 = self.input_tup[1][0], self.input_tup[1][1]

            self.identity_interpreter1 = generate_identity(bx, by1, self.model_kwargs, self.identity, self.identity_kwargs)
            self.identity_interpreter2 = generate_identity(bx, by2, self.model_kwargs, self.identity, self.identity_kwargs)
 
            heatmap1_1, heatmapimg1_1 = self.identity_interpreter1(bx1, by1)  # interpreter1_cls1_input1
            heatmap1_2, heatmapimg1_2 = self.identity_interpreter2(bx1, by2)  # interpreter2_cls2_input1
            heatmap2_1, heatmapimg2_1 = self.identity_interpreter1(bx2, by1)  # interpreter1_cls1_input2
            heatmap2_2, heatmapimg2_2 = self.identity_interpreter2(bx2, by2)  # interpreter2_cls2_input2

        heatmap1 = 0.5 * (heatmap1_1 + heatmap2_1)
        heatmapimg1 =  heatmap1 + np.float32(bx)
        heatmapimg1 = heatmapimg1 / np.max(heatmapimg1)

        heatmap2 = 0.5 * (heatmap1_2 + heatmap2_2)
        heatmapimg2 =  heatmap2 + np.float32(bx)
        heatmapimg2 = heatmapimg2 / np.max(heatmapimg2)

        return heatmap1, heatmapimg1, heatmap2, heatmapimg2 





