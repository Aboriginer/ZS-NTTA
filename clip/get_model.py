from data.dataset_class_names import get_classnames
from .custom_clip import ClipTestTimeTuning


def get_model(args, learned_cls=False):
    classnames = get_classnames(args.data.test_set)
    # most baselines in our framework is based on TPT code framework
    model = ClipTestTimeTuning(args.gpu, classnames, None, arch=args.model.arch,
                            n_ctx=args.model.n_ctx, ctx_init=args.model.ctx_init, learned_cls=learned_cls)

    return model