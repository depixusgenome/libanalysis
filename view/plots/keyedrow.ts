import * as p         from "core/properties"
import {RowView, Row} from "models/layouts/row"
import {ToolbarBox}   from "models/tools/toolbar_box"
import {ToolProxy}    from "models/tools/tool_proxy"
import {Range1d}      from "models/ranges/range1d"
import {Tool}         from "models/tools/tool"
import {Plot}         from "models/plots/plot"

export namespace DpxKeyedRow {
    export type Attrs = p.AttrsOf<Props>

    export type Props = Row.Props & {
        _curr:    p.Property<Tool | null>
        fig:      p.Property<Plot | null>
        toolbar:  p.Property<any>
        keys:     p.Property<{[key: string]: string}>
        zoomrate: p.Property<number>
        panrate:  p.Property<number>
    }
}

export interface DpxKeyedRow extends DpxKeyedRow.Attrs {}

export class DpxKeyedRowView extends RowView {
    model: DpxKeyedRow
    render(): void {
        super.render()
        this.el.setAttribute("tabindex", "1")
        this.el.onkeydown = (evt) => this.model.dokeydown(evt)
        this.el.onkeyup   = () => this.model.dokeyup()
    }
}

export class DpxKeyedRow extends Row {
    properties: DpxKeyedRow.Props
    constructor(attrs?: Partial<DpxKeyedRow.Attrs>) {
        super(attrs);
    }

    _get_tb(): any {
        if(this.toolbar != null)
            return this.toolbar
        if(this.fig != null)
            return this.fig.toolbar
        return null
    }

    _get_tool(name:string): any {
        let tools: {type: string}[] = []
        if (this.toolbar != null) {
            if(
                this.toolbar instanceof ToolbarBox
                && this.toolbar.toolbar.gestures != null
            ) {
                for (let tpe in this.toolbar.toolbar.gestures){
                    let ref1: ToolProxy[] = (this.toolbar.toolbar.gestures as any)[tpe].tools
                    for(let proxy of ref1) {
                        for (let ref2 of proxy.tools) {
                            if (ref2.type === name)
                                return proxy.tools;
                        }
                    }
                }
                return null
            } else
                tools = this.toolbar.tools
        } else if(this.fig != null && this.fig.toolbar.tools != null)
            tools = this.fig.toolbar.tools
        else
            return null

        for (let itm of tools)
            if (itm.type === name)
                return itm;
        return null

    }

    _activate(tool:Tool): void { tool.active = !tool.active }

    _set_active(name:string): void {
        let tool = this._get_tool(name)
        if(tool != null)
        {
            let tbar: any = (this.fig as any).toolbar
            if(this.toolbar instanceof ToolbarBox)
                tbar = this.toolbar.toolbar as any
            else if(this.toolbar != null)
                tbar = this.toolbar as any

            this._curr = tbar.gestures.pan.active as Tool
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
            return rng.bounds as [number, number]
        return [rng.reset_start as number, rng.reset_end as number]
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
        let fig    = this.fig as Plot
        let rng    = fig.x_range as Range1d
        let bounds = this._bounds(rng)

        rng.start = bounds[0]
        rng.end   = bounds[1]

        rng       = fig.y_range as Range1d
        bounds    = this._bounds(rng)
        rng.start = bounds[0]
        rng.end   = bounds[1]
    }

    dokeydown(evt: KeyboardEvent) : void {
        if(this.fig == null)
            return

        let val: string  = ""
        let tmp: {[key: string]: string} = {
            'alt': 'Alt', 
            'shift': 'Shift',
            'ctrl': 'Control',
            'meta': 'Meta'
        }
        for(let name in tmp)
            if((evt as any as {[key: string]: boolean})[name+'Key'])
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
                (this as any)["_do_"+tool](dir, ((this.fig as any)[rng]) as Range1d)
            }
        }
    }

    dokeyup(): void {
        if(this._curr != null)
            this._activate(this._curr)
        this._curr  = null
    }

    static initClass(): void {
        this.prototype.default_view= DpxKeyedRowView
        this.prototype.type= "DpxKeyedRow"
        this.override({css_classes : ["dpx-bk-grid-row"]})
        this.internal({
            _curr:    [p.Any, null]
        })

        this.define<DpxKeyedRow.Props>({
            fig:      [p.Instance, null],
            toolbar:  [p.Instance, null],
            keys:     [p.Any,   {}],
            zoomrate: [p.Number, 0],
            panrate:  [p.Number, 0],
        })
    }
}
DpxKeyedRow.initClass()
