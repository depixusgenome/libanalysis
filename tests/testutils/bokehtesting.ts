import *        as p    from "core/properties"
import {Model}          from "model"
import {DOMView}        from "core/dom_view"
import {MenuItemClick, ButtonClick}  from "core/bokeh_events"
import {Dropdown}       from "models/widgets/dropdown"
import {Button}          from "models/widgets/button"

declare var jQuery: any
declare var Bokeh: any

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
        window.onerror = function(message, source, lineno, colno, error) {
            var elem = document.createElement("p");
            elem.innerHTML = `${source} [${lineno}-${colno}]: ${message} // ${error}`;
            (elem as any).classList.add('dpx-test-error');
            document.body.appendChild(elem);
        };
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
        let root = this._model()
        console.debug("changed attribute: ", root, this.attrs, this.attr, this.value)
        if(root != null)
        {
            let mdl = root
            for(let i in this.attrs)
                mdl = mdl[i]

            if(typeof mdl === undefined)
                throw `Could not find ${this.attrs} in ${root}`;

            if(this.attr == "clicks" && mdl instanceof Button)
            {
                mdl.clicks = mdl.clicks + 1
                mdl.trigger_event(new ButtonClick())
            }
            else if(this.attr == "value" && mdl instanceof Dropdown)
            {
                mdl.trigger_event(new MenuItemClick(this.value))
                mdl.value = this.value
            } else
            {
                mdl[this.attr] = this.value
                if(this.attrs.length == 0)
                {
                    try { root.properties[this.attr].change.emit; }
                    catch (e)
                    { throw new Error(`Missing ${this.attr} in ${root}`); }
                    root.properties[this.attr].change.emit()
                }
                else
                    root.properties[this.attrs[0]].change.emit()
            }
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
        })
    }
}
DpxTestLoaded.initClass()
