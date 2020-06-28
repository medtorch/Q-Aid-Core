from __future__ import absolute_import, division, print_function

import mimetypes
from pathlib import Path

from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from werkzeug import wrappers


class PyTorchXAIPlugin(base_plugin.TBPlugin):
    plugin_name = "pytorchxai"

    def __init__(self, context):
        """Instantiates ExamplePlugin.

        Args:
        context: A base_plugin.TBContext instance.
        """
        plugin_directory_path_part = f"/data/plugin/{self.plugin_name}/"
        self._multiplexer = context.multiplexer
        self._offset_path = len(plugin_directory_path_part)
        self._prefix_path = Path(__file__).parent / 'pytorchxai'

    def is_active(self):
        """Returns whether there is relevant data for the plugin to process.

        When there are no runs with greeting data, TensorBoard will hide the
        plugin from the main navigation bar.
        """
        return bool(
            self._multiplexer.PluginRunToTagToContent(self.plugin_name)
        )

    def get_plugin_apps(self):
        return {
            "/static/*": self._serve_static_file,
        }

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            es_module_path="/static/index.js",
            tab_name="PyTorchXAI"
        )

    @wrappers.Request.application
    def _serve_static_file(self, request):
        static_path_part = request.path[self._offset_path:]
        resource_path = Path(static_path_part)

        if not resource_path.parent != "static":
            return http_util.Respond(
                request, "Resource not found", "text/plain", code=404
            )

        resource_absolute_path = str(self._prefix_path / resource_path)
        with open(resource_absolute_path, "rb") as read_file:
            mimetype = mimetypes.guess_type(resource_absolute_path)[0]
            return http_util.Respond(
                request, read_file.read(), content_type=mimetype
            )
