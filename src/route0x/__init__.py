# route0x/__init__.py

__all__ = ['RouteFinder', 'RouteBuilder', 'Route', 'RouteBuilderRequest']

# Lazy loader for RouteFinder and RouteBuilder
class _LazyLoader:
    def __getattr__(self, name):
        if name == "RouteFinder":
            from .route_finder import RouteFinder
            return RouteFinder
        elif name == "RouteBuilder":
            from .route_builder import RouteBuilder
            return RouteBuilder
        elif name == "Route":
            from .route_builder import Route
            return Route
        elif name == "RouteBuilderRequest":
            from .route_builder import RouteBuilderRequest
            return RouteBuilderRequest
        raise AttributeError(f"Module 'route0x' has no attribute '{name}'")

# Create a single instance of the lazy loader
_lazy_loader = _LazyLoader()

# Expose classes with lazy loading
RouteFinder = _lazy_loader.RouteFinder
RouteBuilder = _lazy_loader.RouteBuilder
Route = _lazy_loader.Route
RouteBuilderRequest = _lazy_loader.RouteBuilderRequest
