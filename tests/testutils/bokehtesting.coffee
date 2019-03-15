import *        as p    from "core/properties"
import {Model}          from "model"
import {DOMView}        from "core/dom_view"

export class DpxTestLoadedView extends DOMView
    className: "dpx-test"

    connect_signals: () ->
        super()
        console.log("DpxTestLoadedView connect_signals")
        @connect(@model.properties.event_cnt.change, () => @model._press())
        @connect(@model.properties.value_cnt.change, () => @model._change())

export class DpxTestLoaded extends Model
    default_view: DpxTestLoadedView
    type: "DpxTestLoaded"
    constructor : (attributes, options) ->
        super(attributes, options)
        $(() => @done = 1)

        self = @

        oldlog       = console.log
        console.log  = () -> self._tostr(oldlog, 'debug', arguments)

        oldinfo      = console.info
        console.info = () -> self._tostr(oldinfo, 'info', arguments)

        oldwarn      = console.warn
        console.warn = () -> self._tostr(oldwarn, 'warn', arguments)

    _tostr: (old, name, args) ->
        old.apply(console, args)

        str = ""
        for i in args
            str = str + " " + i
        @[name] = ""
        @[name] = str

    _create_evt: (name) ->
        evt = $.Event(name)
        evt.altKey   = @event.alt
        evt.shiftKey = @event.shift
        evt.ctrlKey  = @event.ctrl
        evt.metaKey  = @event.meta
        evt.key      = @event.key

        return evt

    _model: () ->
        return Bokeh.documents[0].get_model_by_id(@modelid)

    _press: () ->
        console.debug("pressed key: ", @event.ctrl, @event.key)
        mdl = @_model()
        if mdl?
            mdl.dokeydown?(@_create_evt('keydown'))
            mdl.dokeyup?(@_create_evt('keyup'))
        else
            console.log("pressed key but there's no model")

    _change: () ->
        console.debug("changed attribute: ", @attrs, @value)
        root = @_model()
        if root?
            mdl = root
            for i in @attrs
                mdl = mdl[i]

            mdl[@attr] = @value
            if @attrs.length == 0
                root.properties[@attr].change.emit()
            else
                root.properties[@attrs[0]].change.emit()
        else
            console.log("changed key but there's no model")

    @define {
        done:  [p.Number, 0]
        event: [p.Any,   {}]
        event_cnt: [p.Int, 0]
        modelid: [p.String, '']
        attrs: [p.Array, []]
        attr : [p.String, '']
        value: [p.Any,   {}]
        value_cnt: [p.Int, 0]
        debug: [p.String, '']
        warn: [p.String, '']
        info: [p.String, '']
    }
