from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipelines.pipeline import FaceSwapPipeline

    register(FaceSwapPipeline)
