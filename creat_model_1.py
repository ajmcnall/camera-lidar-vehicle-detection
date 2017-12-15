def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 cancate with new layers trained with current data 

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)
    top16_yolo = Model(yolo_model.input, yolo_model.layers[-17].output)
    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body_topless = Model(model_body.inputs, model_body.layers[-2].output)
            model_body_topless.save_weights(topless_yolo_path)
            model_body_12 = Model(model_body.inputs, model_body.layers[-13].output)
            model_body_12.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)
        top16_yolo.load_weights(topless_yolo_path)
   # trained between 26*26 ->13*13 
    if freeze_body:
        for layer in topless_yolo.layers:     
            layer.trainable = False      
        for layer in top12_yolo.layers:
            layer.trainable = False
    conv31 = compose(
        DarknetConv2D_BN_Leaky(256, (3, 3))
        DarknetConv2D_BN_Leaky(512, (3, 3))
        MaxPooling2D()
        DarknetConv2D_BN_Leaky(1024, (3, 3))
        DarknetConv2D_BN_Leaky(512, (3, 3))
        DarknetConv2D_BN_Leaky(1024, (3, 3))       
        )(top12_yolo.output)
    # combine yolov2 with new layers
    x = concatenate([conv31, topless_yolo.output])
    x = DarknetConv2D_BN_Leaky(1024, (3, 3))(x)
    final_layer = DarknetConv2D(len(anchors)*(5+len(class_names), (1, 1))(x),activation='linear')(x)
    
    
    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model