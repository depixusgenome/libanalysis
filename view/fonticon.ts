import *  as p        from "core/properties"
import {AbstractIcon} from "models/widgets/abstract_icon"
import {WidgetView}   from "models/widgets/widget"

export namespace FontIcon {
    export type Attrs = p.AttrsOf<Props>
    export type Props = AbstractIcon.Props & {iconname: p.Property<string>}
}

export interface FontIcon extends FontIcon.Attrs {}

export class FontIconView extends WidgetView {
    model: FontIcon

    initialize() : void {
        super.initialize()
        this.render()
        this.connect(this.model.change, this.render)
    }

    render() {
        super.render()
        this.el.className = "" // erase all CSS classes if re-rendering
        this.el.classList.add("icon-dpx-"+this.model.iconname)
        return this
    }

    static initClass(): void {
        this.prototype.tagName = "span"
    }
}
FontIconView.initClass()

export class FontIcon extends AbstractIcon {
    properties: FontIcon.Props
    static initClass(): void {
        this.prototype.type         = "FontIcon"
        this.prototype.default_view = FontIconView
        this.define({iconname: [p.String, "cog"]})
    }
}
FontIcon.initClass()
