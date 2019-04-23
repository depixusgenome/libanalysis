import *        as p    from "core/properties"
import {Model}          from "model"
import {DOMView}        from "core/dom_view"

declare function jQuery(...args: any[]): any

export namespace DpxTestLoaded {
    export type Attrs = p.AttrsOf<Props>
    export type Props = Model.Props & {
        done:  p.Property<number>
        event: p.Property<any>
        event_cnt: p.Property<number>
        modelid: p.Property<string>
        attrs: p.Property<string[]>
        attr : p.Property<string>
        value: p.Property<any>
        value_cnt: p.Property<number>
        debug: p.Property<string>
        warn: p.Property<string>
        info: p.Property<string>
    }
}

export interface DpxTestLoaded extends DpxTestLoaded.Attrs {}

export class DpxTestLoadedView extends DOMView {
    model: DpxTestLoaded
    connect_signals(): void {
        super.connect_signals();
        this.connect(
            this.model.properties.event_cnt.change,
            () => { this.model._press() }
        )
        this.connect(
            this.model.properties.value_cnt.change,
            () => { this.model._change() }
        )
    }
}

export class DpxTestLoaded extends Model {
    properties: DpxTestLoaded.Props

    constructor(attrs?:Partial<DpxTestLoaded.Attrs>) {
        super(attrs);
        jQuery(() => this.done = 1)

        let oldlog   = console.log
        console.log  = () => this._tostr(oldlog, 'debug', arguments)

        let oldinfo  = console.info
        console.info = () => this._tostr(oldinfo, 'info', arguments)

        let oldwarn  = console.warn
        console.warn = () => this._tostr(oldwarn, 'warn', arguments)
    }

    _tostr(old, name:string, args): void {
        old.apply(console, args)

        let str = ""
        for(let i in args)
            str = str + " " + i
        this[name] = ""
        this[name] = str
    }

    _create_evt(name:string) {
        let evt = jQuery.Event(name)
        evt.altKey   = this.event.alt
        evt.shiftKey = this.event.shift
        evt.ctrlKey  = this.event.ctrl
        evt.metaKey  = this.event.meta
        evt.key      = this.event.key

        return evt
    }

    _model()
    {
        return Bokeh.documents[0].get_model_by_id(this.modelid)
    }

    _press(){
        console.debug("pressed key: ", this.event.ctrl, this.event.key)
        let mdl = this._model()
        if(mdl !== null) {
            if(typeof mdl.dokeydown === "function")
                mdl.dokeydown(this._create_evt('keydown'));
            if(typeof mdl.dokeyup === "function")
                mdl.dokeyup(this._create_evt('keyup'));
        } else
            console.log("pressed key but there's no model")
    }

    _change() {
        console.debug("changed attribute: ", this.attrs, this.value)
        let root = this._model()
        if(root != null)
        {
            let mdl = root
            for(let i in this.attrs)
                mdl = mdl[i]

            mdl[this.attr] = this.value
            if(this.attrs.length == 0)
                root.properties[this.attr].change.emit()
            else
                root.properties[this.attrs[0]].change.emit()
        } else
            console.log("changed key but there's no model")
    }

    static initClass(): void {
        this.prototype.default_view = DpxTestLoadedView
        this.prototype.type         = "DpxTestLoaded"
        this.define<DpxTestLoaded.Props>({
            done:  [p.Number, 0],
            event: [p.Any,   {}],
            event_cnt: [p.Int, 0],
            modelid: [p.String, ''],
            attrs: [p.Array, []],
            attr : [p.String, ''],
            value: [p.Any,   {}],
            value_cnt: [p.Int, 0],
            debug: [p.String, ''],
            warn: [p.String, ''],
            info: [p.String, ''],
        })
    }
}
DpxTestLoaded.initClass()
