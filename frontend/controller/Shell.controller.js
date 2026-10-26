sap.ui.define([
  "sap/ui/core/mvc/Controller"
], function (Controller) {
  "use strict";

  return Controller.extend("ask.datasphere.controller.Shell", {
    onInit: function () {
      var oRouter = this.getOwnerComponent().getRouter();
      oRouter.initialize();
    }
  });
});
