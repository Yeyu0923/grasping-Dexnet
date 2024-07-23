from inference.grasp_generator import GraspGenerator

if __name__ == '__main__':
    generator = GraspGenerator(
        cam_id=218722271289,
        saved_model_path='/home/xfy/GR-ConvNet/robotic-grasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98',
        visualize=True
    )
    generator.load_model()
    generator.run()
