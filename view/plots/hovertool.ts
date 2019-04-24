import * as p  from "core/properties"
import {HoverTool, HoverToolView}              from "models/tools/inspectors/hover_tool"
import {GlyphRenderer}                         from "models/renderers/glyph_renderer"
import {Renderer, RendererView}                from "models/renderers/renderer"
import {Geometry, PointGeometry, SpanGeometry} from "core/geometry"

export namespace DpxHoverTool {
    export type Attrs = p.AttrsOf<Props>
    export type Props = HoverTool.Props & { maxcount: p.Property<number>}
}

export interface DpxHoverTool extends DpxHoverTool.Attrs {}

export class DpxHoverToolView extends HoverToolView {
    model: DpxHoverTool
    _update([renderer_view, {geometry}]: [RendererView, {geometry: PointGeometry | SpanGeometry}]): void {
        if (this.model.active) {
            super._update([renderer_view, {geometry}])
            let ttip = this.ttmodels[renderer_view.model.id]
            if(ttip.data.length > this.model.maxcount) {
                let ind   = Math.floor((ttip.data.length-this.model.maxcount)/2)
                ttip.data = ttip.data.slice(ind, ind + this.model.maxcount)
            }
        }
    }
}

export class DpxHoverTool extends HoverTool {
    properties: DpxHoverTool.Props
    constructor(attrs?: Partial<DpxHoverTool.Attrs>) { super(attrs); }
    static initClass(): void {
        this.prototype.default_view = DpxHoverToolView
        this.define({ maxcount: [ p.Int, 5] })
    }
}
DpxHoverTool.initClass()
