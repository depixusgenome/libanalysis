import * as p         from "core/properties"
import {RowView, Row} from "models/layouts/row"
import {ToolbarBox}   from "models/tools/toolbar_box"
import {Range1d}      from "models/ranges/range1d"
import {Tool}         from "models/tools/tool"
import {Plot}         from "models/plots/plot"

export namespace DpxKeyedRow {
    export type Attrs = p.AttrsOf<Props>

    export type Props = Row.Props & {
        _curr:    p.Property<Tool | null>
        fig:      p.Property<Plot | null>
        toolbar:  p.Property<any>
        keys:     p.Property<string[]>
        zoomrate: p.Property<number>
        panrate:  p.Property<number>
    }
}

export interface DpxKeyedRow extends DpxKeyedRow.Attrs {}

export class DpxKeyedRowView extends RowView {
    model: DpxKeyedRow
    static initClass(): void { this.prototype.className = "dpx-bk-grid-row" }
    render(): void {
        super.render()
        this.el.setAttribute("tabindex", 1)
        this.el.onkeydown = (evt) => this.model.dokeydown(evt)
        this.el.onkeyup   = (evt) => this.model.dokeyup(evt)
    }
}
DpxKeyedRowView.initClass()

export class DpxKeyedRow extends Row {
    properties: DpxKeyedRow.Props
    constructor(attrs?: Partial<DpxKeyedRow.Attrs>) {
        super(attrs);
    }

    _get_tb(): any {
        if(this.toolbar != null)
            return this.toolbar
        return this.fig.toolbar
    }

    _get_tool(name:string): any {
        let tools = null, i: number = 0, len: number = 0
        if (this.toolbar != null) {
            if(this.toolbar instanceof ToolbarBox) {
                let ref = this.toolbar.toolbar.gestures
                for (let _ in ref) {
                    let ref1 = ref[_].tools;
                    for(i = 0, len = ref1.length; i < len; i++) {
                        let proxy = ref1[i];
                        let ref2  = proxy.tools;
                        for (let j = 0, len1 = ref2.length; j < len1; j++) {
                            if (ref2[j].type === name)
                                return proxy;
                        }
                    }
                }
                return null
            } else
                tools = this.toolbar.tools
        } else
            tools = this.fig.toolbar.tools

        for (let i = 0, len = tools.length; i < len; i++) {
            if (tools[i].type === name)
                return tools[i];
        }
        return null

    }

    _activate(tool:Tool): void { tool.active = !tool.active }

    _set_active(name:string): void {
        let tool = this._get_tool(name)
        if(tool != null)
        {
            let tbar = this.fig.toolbar
            if(this.toolbar instanceof ToolbarBox)
                tbar = this.toolbar.toolbar
            else if(this.toolbar != null)
                tbar = this.toolbar

            this._curr = tbar.gestures.pan.active
            if(this._curr !== tool) {
                if(this._curr != null)
                    this._activate(this._curr)

                this._activate(tool)
            }
            else
                this._curr = null
        }
    }

    _bounds(rng: Range1d) : [number, number] {
        if(rng.bounds != null)
            return rng.bounds
        return [rng.reset_start, rng.reset_end]
    }

    _do_zoom(zoomin:boolean, rng: Range1d): void {
        let center = (rng.end+rng.start)*.5
        let delta  = rng.end-rng.start
        if(zoomin)
            delta  /= this.zoomrate
        else
            delta  *= this.zoomrate

        rng.start = center - delta*.5
        rng.end   = center + delta*.5

        let bounds: [number, number] = this._bounds(rng)
        if(bounds[0] > rng.start)
            rng.start = bounds[0]
        if(bounds[1] < rng.end)
            rng.end   = bounds[1]
    }

    _do_pan(panlow:boolean, rng: Range1d): void {
        let bounds: [number, number] = this._bounds(rng)
        let delta: number  = (rng.end-rng.start)*this.panrate*(panlow? -1 : 1)
        if(bounds[0] > rng.start + delta)
            delta = bounds[0]-rng.start
        if(bounds[1] < rng.end   + delta)
            delta = bounds[1]-rng.end

        rng.start = rng.start + delta
        rng.end   = rng.end   + delta
    }

    _do_reset(): void {
        let fig    = this.fig
        let rng    = fig.x_range
        let bounds = this._bounds(rng)

        rng.start = bounds[0]
        rng.end   = bounds[1]

        rng       = fig.y_range
        bounds    = this._bounds(rng)
        rng.start = bounds[0]
        rng.end   = bounds[1]
    }

    dokeydown(evt: KeyboardEvent) : void {
        if(this.fig == null)
            return

        let val: string  = ""
        let tmp = {'alt': 'Alt', 'shift': 'Shift', 'ctrl': 'Control', 'meta': 'Meta'}
        for(let name in tmp)
            if(evt[name+'Key'])
                val += tmp[name]+"-"
        if (val == (evt.key+"-"))
            val = evt.key
        else
            val = val + evt.key

        if(val in this.keys) {
            evt.preventDefault()
            evt.stopPropagation()
            val = this.keys[val]
            if(val == "reset")
                this._do_reset()

            else if(val == "zoom")
                this._set_active("BoxZoomTool")

            else if(val == "pan")
                this._set_active("PanTool")

            else {
                let tool = val.slice(0, 3) === "pan" ? "pan" : "zoom";  
                let rng  = val.indexOf("x") >= 0 ? "x_range" : "y_range";
                let dir  = "low" === val.slice(val.length - 3, val.length);
                this["_do_"+tool](dir, ((this.fig as any)[rng]) as Range1d)
            }
        }
    }

    dokeyup(evt: KeyboardEvent): void {
        if(this._curr != null)
            this._activate(this._curr)
        this._curr  = null
    }

    static initClass(): void {
        this.prototype.default_view= DpxKeyedRowView
        this.prototype.type= "DpxKeyedRow"
        this.internal({
            _curr:    [p.Any, null]
        })

        this.define<DpxKeyedRow.Props>({
            fig:      [p.Instance ],
            toolbar:  [p.Instance, null],
            keys:     [p.Any,   {}],
            zoomrate: [p.Number, 0],
            panrate:  [p.Number, 0],
        })
    }
}
DpxKeyedRow.initClass()
