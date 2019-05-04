import *  as p        from "core/properties"
import {AbstractIcon, AbstractIconView} from "models/widgets/abstract_icon"

export namespace FontIcon {
    export type Attrs = p.AttrsOf<Props>
    export type Props = AbstractIcon.Props & {iconname: p.Property<string>}
}

export interface FontIcon extends FontIcon.Attrs {}

export class FontIconView extends AbstractIconView {
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
    constructor(attrs?:Partial<FontIcon.Attrs>) { super(attrs); }
    static initClass(): void {
        this.prototype.type         = "FontIcon"
        this.prototype.default_view = FontIconView
        this.define<FontIcon.Props>({iconname: [p.String, "cog"]})
    }
}
FontIcon.initClass()
