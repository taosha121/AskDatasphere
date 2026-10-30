sap.ui.define([
  "sap/ui/core/mvc/Controller",
  "sap/ui/model/json/JSONModel",
  "sap/m/Dialog",
  "sap/m/Button",
  "sap/m/Label",
  "sap/m/Text",
  "sap/m/VBox",
  "sap/m/ScrollContainer",
  "sap/m/MessageToast"
], function (
  Controller, JSONModel,
  Dialog, Button, Label, Text, VBox, ScrollContainer, MessageToast
) {
  "use strict";

  var API_BASE = "/api";

  return Controller.extend("ask.datasphere.controller.Chunks", {

    onInit: function () {
      this._oModel = new JSONModel({ items: [], total: 0 });
      this.getView().setModel(this._oModel, "chunks");

      fetch(API_BASE + "/chunks?offset=0&limit=1")
        .then(function (r) { return r.json(); })
        .then(function (data) {
          this._total = data.total;
          this.byId("totalText").setText("Total: " + data.total + " records");
          this._fetchAndRender(0, 20);
        }.bind(this))
        .catch(function (err) {
          this.byId("totalText").setText("Total: unknown");
          MessageToast.show("Init failed: " + err.message);
        }.bind(this));
    },

    onNavToChat: function () {
      this.getOwnerComponent().getRouter().navTo("chat");
    },

    onQuery: function () {
      var offset = parseInt(this.byId("inputStart").getValue(), 10);
      var end    = parseInt(this.byId("inputEnd").getValue(), 10);

      if (isNaN(offset) || isNaN(end)) {
        MessageToast.show("Please enter numeric values.");
        return;
      }
      if (offset < 0 || end < offset) {
        MessageToast.show("Start must be >= 0 and <= End.");
        return;
      }
      var limit = end - offset + 1;
      if (limit > 100) {
        MessageToast.show("Range too large — max 100 per query.");
        return;
      }
      this._fetchAndRender(offset, limit);
    },

    _fetchAndRender: function (offset, limit) {
      var oTable = this.byId("chunkTable");
      oTable.setBusy(true);

      fetch(API_BASE + "/chunks?offset=" + offset + "&limit=" + limit)
        .then(function (r) {
          if (!r.ok) { throw new Error("HTTP " + r.status); }
          return r.json();
        })
        .then(function (data) {
          // Pre-compute display fields and bind directly to the view
          var items = data.items.map(function (item, idx) {
            return {
              globalIdx:      offset + idx,
              deliverableLabel: item.deliverable.replace(/_/g, " "),
              topic_title:    item.topic_title,
              chunk_index:    item.chunk_index,
              preview:        item.text.length > 180
                                ? item.text.substring(0, 180) + "…"
                                : item.text,
              // full data kept for the detail Dialog
              _full: item
            };
          });
          this._oModel.setProperty("/items", items);
          oTable.setBusy(false);
        }.bind(this))
        .catch(function (err) {
          oTable.setBusy(false);
          MessageToast.show("Load failed: " + err.message);
        });
    },

    onShowFullText: function (oEvent) {
      var oCtx  = oEvent.getSource().getParent().getBindingContext("chunks");
      var oData = oCtx.getProperty("_full");
      var idx   = oCtx.getProperty("globalIdx");

      var oDialog = new Dialog({
        title: "#" + idx + " · " + (oData.topic_title || "Full Text"),
        contentWidth: "62rem",
        contentHeight: "34rem",
        resizable: true,
        draggable: true,
        content: [
          new VBox({
            renderType: "Bare",
            items: [
              new Label({ text: "Deliverable", design: "Bold" }),
              new Text({ text: oData.deliverable.replace(/_/g, " ") }),
              new Label({ text: "Source File", design: "Bold" }).addStyleClass("sapUiTinyMarginTop"),
              new Text({ text: oData.source_file }),
              new Label({ text: "Chunk Index", design: "Bold" }).addStyleClass("sapUiTinyMarginTop"),
              new Text({ text: String(oData.chunk_index) }),
              new Label({ text: "Full Text", design: "Bold" }).addStyleClass("sapUiTinyMarginTop"),
              new ScrollContainer({
                height: "22rem",
                vertical: true,
                content: [new Text({ text: oData.text, wrapping: true })]
              })
            ]
          }).addStyleClass("sapUiSmallMargin")
        ],
        beginButton: new Button({
          text: "Close",
          press: function () { oDialog.close(); }
        }),
        afterClose: function () { oDialog.destroy(); }
      });

      this.getView().addDependent(oDialog);
      oDialog.open();
    }

  });
});
