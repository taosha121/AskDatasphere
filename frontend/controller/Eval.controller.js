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
], function (Controller, JSONModel, Dialog, Button, Label, Text, VBox, ScrollContainer, MessageToast) {
  "use strict";

  var METRICS = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"];

  function scoreState(v) {
    if (v == null || isNaN(v)) return "None";
    return v >= 0.75 ? "Success" : v >= 0.50 ? "Warning" : "Error";
  }

  function scoreLabel(v) {
    return (v == null || isNaN(v)) ? "—" : v.toFixed(3);
  }

  // 流式解析 CSV，正确处理字段内嵌换行（retrieved_contexts 里有大量 \n）
  function parseCSV(text) {
    var rows = [], headers = null;
    var fields = [], cur = "", inQ = false;

    for (var i = 0; i <= text.length; i++) {
      var c = i < text.length ? text[i] : "\n"; // 末尾哨兵，确保最后一行被 flush
      if (c === '"') {
        if (inQ && text[i + 1] === '"') { cur += '"'; i++; }
        else { inQ = !inQ; }
      } else if (c === ',' && !inQ) {
        fields.push(cur); cur = "";
      } else if ((c === '\n' || c === '\r') && !inQ) {
        if (c === '\r' && text[i + 1] === '\n') i++;
        if (cur || fields.length) {
          fields.push(cur); cur = "";
          if (!headers) {
            headers = fields.map(function (h) { return h.trim(); });
          } else {
            var obj = {};
            headers.forEach(function (h, idx) {
              obj[h] = fields[idx] !== undefined ? fields[idx] : "";
            });
            rows.push(obj);
          }
          fields = [];
        }
      } else {
        cur += c;
      }
    }
    return rows;
  }

  return Controller.extend("ask.datasphere.controller.Eval", {

    onInit: function () {
      this._oModel = new JSONModel({ summary: {}, rows: [] });
      this.getView().setModel(this._oModel, "eval");
      this._rawRows = [];
      this._sortKey = "context_recall";
      this._sortAsc = true;
    },

    onNavToChat: function () {
      this.getOwnerComponent().getRouter().navTo("chat");
    },

    onFileChange: function (oEvent) {
      this._file = oEvent.getParameter("files")[0];
    },

    onAnalyze: function () {
      if (!this._file) {
        MessageToast.show("Please select a CSV file first.");
        return;
      }
      var that = this;
      var reader = new FileReader();
      reader.onload = function (e) {
        try {
          that._processCSV(e.target.result);
        } catch (err) {
          MessageToast.show("Parse error: " + err.message);
        }
      };
      reader.readAsText(this._file);
    },

    _processCSV: function (text) {
      var parsed = parseCSV(text);
      if (!parsed.length) {
        MessageToast.show("No data rows found.");
        return;
      }

      // 计算每行的展示字段
      var rows = parsed.map(function (r, i) {
        var row = { _idx: i };
        METRICS.forEach(function (m) {
          var v = parseFloat(r[m]);
          row[m] = isNaN(v) ? null : v;
          row[m + "_label"] = scoreLabel(row[m]);
          row[m + "_state"] = scoreState(row[m]);
        });
        // question 来自 user_input 列（RAGAS 输出格式）
        var q = r["user_input"] || r["question"] || "";
        row["question_full"] = q;
        row["question_preview"] = q.length > 90 ? q.substring(0, 90) + "…" : q;
        row["answer"]        = r["response"] || r["answer"] || "";
        row["ground_truth"]  = r["reference"] || r["ground_truth"] || "";
        return row;
      });

      this._rawRows = rows;
      this._applySort();
      this._buildSummary(rows);

      this.byId("summarySection").setVisible(true);
      this.byId("tableSection").setVisible(true);
      this.byId("sampleCount").setText(rows.length + " questions evaluated");
    },

    _buildSummary: function (rows) {
      var summary = {};
      METRICS.forEach(function (m) {
        var vals = rows.map(function (r) { return r[m]; }).filter(function (v) { return v != null; });
        var avg = vals.length ? vals.reduce(function (a, b) { return a + b; }, 0) / vals.length : null;
        summary[m + "_label"] = scoreLabel(avg);
        summary[m + "_state"] = scoreState(avg);
        summary[m + "_pct"]   = avg != null ? Math.round(avg * 100) : 0;
      });
      this._oModel.setProperty("/summary", summary);
    },

    _applySort: function () {
      var key = this._sortKey;
      var asc = this._sortAsc;
      var sorted = this._rawRows.slice().sort(function (a, b) {
        var va = a[key] != null ? a[key] : (asc ? Infinity : -Infinity);
        var vb = b[key] != null ? b[key] : (asc ? Infinity : -Infinity);
        return asc ? va - vb : vb - va;
      });
      sorted.forEach(function (r, i) { r._rank = i + 1; });
      this._oModel.setProperty("/rows", sorted);
    },

    onSortChange: function (oEvent) {
      this._sortKey = oEvent.getParameter("selectedItem").getKey();
      this._applySort();
    },

    onSortDirChange: function (oEvent) {
      this._sortAsc = oEvent.getParameter("pressed");
      this._applySort();
    },

    onRowPress: function (oEvent) {
      var oCtx  = oEvent.getSource().getBindingContext("eval");
      var oData = oCtx.getObject();

      var metricRows = METRICS.map(function (m) {
        return new VBox({
          renderType: "Bare",
          items: [
            new Label({ text: m.replace(/_/g, " "), design: "Bold" }).addStyleClass("sapUiTinyMarginTop"),
            new Text({ text: oData[m + "_label"] + "  (" + oData[m + "_state"] + ")" })
          ]
        });
      });

      var oDialog = new Dialog({
        title: "#" + oData._rank + " · Question Detail",
        contentWidth: "56rem",
        contentHeight: "36rem",
        resizable: true,
        draggable: true,
        content: [
          new ScrollContainer({
            height: "34rem",
            vertical: true,
            content: [
              new VBox({
                renderType: "Bare",
                items: [
                  new Label({ text: "Question", design: "Bold" }),
                  new Text({ text: oData.question_full, wrapping: true }),
                  new Label({ text: "Answer", design: "Bold" }).addStyleClass("sapUiSmallMarginTop"),
                  new Text({ text: oData.answer, wrapping: true }),
                  new Label({ text: "Ground Truth", design: "Bold" }).addStyleClass("sapUiSmallMarginTop"),
                  new Text({ text: oData.ground_truth, wrapping: true }),
                  new Label({ text: "Scores", design: "Bold" }).addStyleClass("sapUiSmallMarginTop")
                ].concat(metricRows)
              }).addStyleClass("sapUiSmallMargin")
            ]
          })
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
