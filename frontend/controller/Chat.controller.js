sap.ui.define([
  "sap/ui/core/mvc/Controller",
  "sap/ui/model/json/JSONModel",
  "sap/m/Dialog",
  "sap/m/Button",
  "sap/m/Label",
  "sap/m/Text",
  "sap/m/VBox",
  "sap/m/ScrollContainer",
  "sap/m/MessageToast",
  "sap/m/BusyDialog"
], function (
  Controller, JSONModel,
  Dialog, Button, Label, Text, VBox, ScrollContainer, MessageToast, BusyDialog
) {
  "use strict";

  return Controller.extend("ask.datasphere.controller.Chat", {

    onInit: function () {
      this._oModel = new JSONModel({ items: [] });
      this.getView().setModel(this._oModel, "chat");
    },

    onNavToChunks: function () {
      this.getOwnerComponent().getRouter().navTo("chunks");
    },

    onNavToEval: function () {
      this.getOwnerComponent().getRouter().navTo("eval");
    },

    onAsk: function () {
      var question = this.byId("questionInput").getValue().trim();
      if (!question) {
        MessageToast.show("Please enter a question.");
        return;
      }

      var topK = this.byId("topKInput").getValue();

      this.byId("askBtn").setEnabled(false);
      this.byId("answerPanel").setVisible(false);
      this.byId("promptPanel").setVisible(false);
      this._oModel.setProperty("/items", []);
      this.byId("chunksTitle").setText("Retrieved Chunks (Top " + topK + ")");

      var oBusy = new BusyDialog({ text: "Querying…" });
      oBusy.open();

      fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question, top_k: topK })
      })
        .then(function (r) {
          if (!r.ok) { throw new Error("HTTP " + r.status); }
          return r.json();
        })
        .then(function (data) {
          oBusy.close();
          this.byId("askBtn").setEnabled(true);

          // Right column: chunks
          var items = data.chunks.map(function (c, i) {
            var score = c.score != null ? c.score : null;
            var scoreLabel = score != null ? score.toFixed(4) : "—";
            var scoreState = score == null ? "None"
              : score >= 0.75 ? "Success"
              : score >= 0.50 ? "Warning"
              : "Error";
            return {
              rank:        i + 1,
              deliverable: c.deliverable.replace(/_/g, " "),
              topic_title: c.topic_title,
              chunk_index: c.chunk_index,
              score:       score,
              scoreLabel:  scoreLabel,
              scoreState:  scoreState,
              preview:     c.text.length > 300 ? c.text.substring(0, 300) + "…" : c.text,
              _full:       c
            };
          });
          this._oModel.setProperty("/items", items);

          // Right column: prompt
          this.byId("promptText").setValue(data.prompt);
          this.byId("promptPanel").setVisible(true);

          // Left column: answer
          this.byId("answerText").setText(data.answer);
          this.byId("answerPanel").setVisible(true);
        }.bind(this))
        .catch(function (err) {
          oBusy.close();
          this.byId("askBtn").setEnabled(true);
          MessageToast.show("Error: " + err.message);
        }.bind(this));
    },

    onShowChunk: function (oEvent) {
      var oCtx  = oEvent.getSource().getParent().getParent().getBindingContext("chat");
      var oData = oCtx.getProperty("_full");
      var rank  = oCtx.getProperty("rank");

      var oDialog = new Dialog({
        title: "#" + rank + " · " + (oData.topic_title || "Full Text"),
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
